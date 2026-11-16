from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw

from .constants import COCO_CATEGORY_ID_TO_INDEX, COCO_CLASSES, IGNORE_INDEX
from .labels import build_coco_cat_id_to_index, load_coco_json
from .utils import to_uint8_image

try:
    from pycocotools import mask as _coco_mask_api
except ImportError:
    _coco_mask_api = None


class VOCClassificationDataset(Dataset):
    def __init__(self, voc_root: Path, labels_df: pd.DataFrame, classes: tuple[str, ...], transform) -> None:
        self.voc_root = Path(voc_root)
        self.image_dir = self.voc_root / "JPEGImages"
        self.classes = tuple(classes)
        self.image_ids = labels_df["image_id"].tolist()
        self.targets = torch.tensor(labels_df[list(self.classes)].to_numpy(dtype="float32"))
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def get_image_path(self, index: int) -> Path:
        return self.image_dir / f"{self.image_ids[index]}.jpg"

    def load_gt_mask(self, index: int, size: int | tuple[int, int] | None = None) -> np.ndarray:
        return load_gt_mask(self.voc_root, self.image_ids[index], size=size)

    def __getitem__(self, index):
        image_path = self.get_image_path(index)
        with Image.open(image_path) as source:
            image = self.transform(source.convert("RGB"))
        return image, self.targets[index]


def detect_coco_year(coco_root: Path | str) -> str:
    """Detect whether `coco_root` contains COCO 2017 or COCO 2014 files."""
    root = Path(coco_root)
    for year in ("2017", "2014"):
        ann_dir = root / "annotations"
        if (
            (ann_dir / f"instances_train{year}.json").is_file()
            or (ann_dir / f"instances_val{year}.json").is_file()
            or (root / f"train{year}").is_dir()
            or (root / "images" / f"train{year}").is_dir()
        ):
            return year
    raise FileNotFoundError(
        f"Cannot detect COCO 2017 or 2014 layout under {root} "
        "(expected annotations/instances_{train,val}{2017,2014}.json)."
    )


def _normalize_coco_split(coco_root: Path, split: str) -> str:
    if split in ("train", "val"):
        return f"{split}{detect_coco_year(coco_root)}"
    return split


def _find_coco_image_dir(coco_root: Path, split: str) -> Path:
    candidates = [
        coco_root / split,
        coco_root / "images" / split,
        coco_root / "images",
        coco_root / "JPEGImages",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"COCO image directory for split {split!r} not found. Checked: {searched}")


def resolve_coco_paths(
    coco_root: Path | str,
    split: str = "train2017",
    image_dir: Path | str | None = None,
    ann_file: Path | str | None = None,
) -> tuple[Path, Path]:
    """Resolve `(image_dir, ann_file)` for COCO 2014/2017 layouts."""
    root = Path(coco_root)
    norm_split = _normalize_coco_split(root, split)
    resolved_img_dir = Path(image_dir) if image_dir is not None else _find_coco_image_dir(root, norm_split)
    resolved_ann_file = (
        Path(ann_file)
        if ann_file is not None
        else root / "annotations" / f"instances_{norm_split}.json"
    )
    if not resolved_ann_file.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {resolved_ann_file}")
    return resolved_img_dir, resolved_ann_file


def _decode_coco_rle_string(counts_str: str | bytes) -> list[int]:
    """Decode COCO's LEB128 delta-encoded compressed RLE string to raw run lengths."""
    if isinstance(counts_str, bytes):
        counts_str = counts_str.decode("ascii")
    counts: list[int] = []
    pos = 0
    n = len(counts_str)
    while pos < n:
        x = 0
        k = 0
        more = True
        while more:
            c = ord(counts_str[pos]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            pos += 1
            k += 1
            if not more and (c & 0x10):
                x -= 1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_coco_rle(rle: dict, height: int | None = None, width: int | None = None) -> np.ndarray:
    """Decode a COCO compressed or uncompressed RLE dictionary into a `(H, W)` bool mask."""
    size = rle.get("size")
    h = int(size[0]) if size is not None else int(height)
    w = int(size[1]) if size is not None else int(width)
    counts = rle["counts"]

    if _coco_mask_api is not None:
        if isinstance(counts, list):
            encoded = _coco_mask_api.frPyObjects(rle, h, w)
        else:
            encoded = {"size": [h, w], "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}
        return _coco_mask_api.decode(encoded).astype(bool)

    run_lengths = _decode_coco_rle_string(counts) if isinstance(counts, (str, bytes)) else counts
    flat = np.zeros(h * w, dtype=bool)
    idx = 0
    fg = False
    for length in run_lengths:
        length_i = int(length)
        if fg and length_i > 0:
            flat[idx : idx + length_i] = True
        idx += length_i
        fg = not fg
    return flat.reshape((h, w), order="F")


def decode_coco_ann_mask(ann: dict, height: int, width: int) -> np.ndarray:
    """Decode one COCO instance annotation (polygon list or RLE dict) into a `(H, W)` bool mask."""
    seg = ann["segmentation"]
    if isinstance(seg, dict):
        return decode_coco_rle(seg, height=height, width=width)

    if isinstance(seg, list):
        if not seg:
            return np.zeros((height, width), dtype=bool)
        if _coco_mask_api is not None:
            rles = _coco_mask_api.frPyObjects(seg, height, width)
            if isinstance(rles, list):
                decoded = _coco_mask_api.decode(_coco_mask_api.merge(rles))
            else:
                decoded = _coco_mask_api.decode(rles)
            return decoded.astype(bool)

        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for poly in seg:
            if len(poly) >= 6:
                draw.polygon(poly, fill=1)
        return np.asarray(canvas, dtype=bool)

    raise TypeError(f"Unsupported COCO segmentation format: {type(seg).__name__}")


def index_coco_annotations(
    ann_file: Path | str,
    target_image_ids: set[int] | None = None,
    coco_data: dict | None = None,
) -> tuple[dict[int, list[dict]], list[dict]]:
    """Index COCO annotations by `image_id` (optionally filtered to `target_image_ids`)."""
    data = coco_data if coco_data is not None else load_coco_json(ann_file)
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", ()):
        img_id = int(ann["image_id"])
        if target_image_ids is not None and img_id not in target_image_ids:
            continue
        anns_by_image.setdefault(img_id, []).append(ann)
    return anns_by_image, list(data.get("categories", ()))


def build_coco_semantic_mask(
    anns: list[dict],
    height: int,
    width: int,
    cat_id_to_index: dict[int, int] = COCO_CATEGORY_ID_TO_INDEX,
    ignore_crowd: bool = False,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Rasterize COCO instance annotations into an `(H, W)` int64 semantic mask (`0`=bg, `1..K`=classes)."""
    mask = np.zeros((height, width), dtype=np.int64)
    if not anns:
        return mask

    # Paint larger instances first so smaller foreground objects sitting in front remain visible.
    ordered = sorted(anns, key=lambda a: float(a.get("area", 0.0)), reverse=True)
    for ann in ordered:
        cls_idx = cat_id_to_index.get(int(ann["category_id"]))
        if cls_idx is None:
            continue
        if ignore_crowd and int(ann.get("iscrowd", 0)) == 1:
            paint_val = ignore_index
        else:
            paint_val = cls_idx + 1
        ann_mask = decode_coco_ann_mask(ann, height=height, width=width)
        mask[ann_mask] = paint_val
    return mask


def load_coco_gt_mask(
    image_id: int,
    height: int,
    width: int,
    anns: list[dict] | None = None,
    cat_id_to_index: dict[int, int] = COCO_CATEGORY_ID_TO_INDEX,
    size: int | tuple[int, int] | None = None,
    mask_dir: Path | str | None = None,
    file_name: str | None = None,
    ignore_crowd: bool = False,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Load a COCO semantic GT mask from `mask_dir` (if present) or decode from COCO annotations."""
    if mask_dir is not None:
        m_dir = Path(mask_dir)
        candidates = []
        if file_name is not None:
            candidates.append(m_dir / f"{Path(file_name).stem}.png")
        candidates.extend([m_dir / f"{int(image_id):012d}.png", m_dir / f"{image_id}.png"])
        for cand in candidates:
            if cand.is_file():
                with Image.open(cand) as gt_img:
                    if size is not None:
                        target = (size, size) if isinstance(size, int) else tuple(size)
                        gt_img = gt_img.resize(target, resample=Image.NEAREST)
                    return np.array(gt_img, dtype=np.int64)

    mask = build_coco_semantic_mask(
        anns or [],
        height=height,
        width=width,
        cat_id_to_index=cat_id_to_index,
        ignore_crowd=ignore_crowd,
        ignore_index=ignore_index,
    )
    if size is not None:
        target = (size, size) if isinstance(size, int) else tuple(size)
        if (width, height) != target:
            mask_u8 = Image.fromarray(mask.astype(np.uint8)).resize(target, resample=Image.NEAREST)
            return np.array(mask_u8, dtype=np.int64)
    return mask


class COCOClassificationDataset(Dataset):
    """Multi-label image classification + optional semantic GT mask dataset for MS COCO 2014/2017."""

    def __init__(
        self,
        coco_root: Path | str,
        labels_df: pd.DataFrame,
        classes: tuple[str, ...] = COCO_CLASSES,
        transform=None,
        split: str = "train2017",
        image_dir: Path | str | None = None,
        ann_file: Path | str | None = None,
        mask_dir: Path | str | None = None,
        coco_anns_by_image: dict[int, list[dict]] | None = None,
        categories: list[dict] | None = None,
        ignore_crowd: bool = False,
    ) -> None:
        self.coco_root = Path(coco_root)
        self.split = _normalize_coco_split(self.coco_root, split) if split in ("train", "val") else split
        self.image_dir = Path(image_dir) if image_dir is not None else _find_coco_image_dir(self.coco_root, self.split)
        if ann_file is not None:
            self.ann_file: Path | None = Path(ann_file)
        else:
            default_ann = self.coco_root / "annotations" / f"instances_{self.split}.json"
            self.ann_file = default_ann if default_ann.is_file() else None
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.ignore_crowd = ignore_crowd

        self.classes = tuple(classes)
        self.image_ids = [int(i) for i in labels_df["image_id"].tolist()]
        if "file_name" in labels_df.columns:
            self.file_names = labels_df["file_name"].tolist()
        elif "2014" in self.split:
            self.file_names = [f"COCO_{self.split}_{img_id:012d}.jpg" for img_id in self.image_ids]
        else:
            self.file_names = [f"{img_id:012d}.jpg" for img_id in self.image_ids]
        self.widths = labels_df["width"].astype(int).tolist() if "width" in labels_df.columns else None
        self.heights = labels_df["height"].astype(int).tolist() if "height" in labels_df.columns else None
        self.targets = torch.tensor(labels_df[list(self.classes)].to_numpy(dtype="float32"))
        self.transform = transform

        self._anns_by_image = coco_anns_by_image
        self._cat_id_to_index = build_coco_cat_id_to_index(categories, classes=self.classes) if (
            categories is not None or self.classes == COCO_CLASSES
        ) else None

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_image_path(self, index: int) -> Path:
        return self.image_dir / self.file_names[index]

    def _ensure_anns_loaded(self) -> None:
        if self._anns_by_image is not None:
            return
        if self.ann_file is None or not self.ann_file.is_file():
            raise FileNotFoundError(
                "COCO annotation JSON is required to decode GT masks when `mask_dir` has no PNG for the image."
            )
        anns_by_image, categories = index_coco_annotations(
            self.ann_file, target_image_ids=set(self.image_ids)
        )
        self._anns_by_image = anns_by_image
        if self._cat_id_to_index is None:
            self._cat_id_to_index = build_coco_cat_id_to_index(categories, classes=self.classes)

    def load_gt_mask(self, index: int, size: int | tuple[int, int] | None = None) -> np.ndarray:
        img_id = self.image_ids[index]
        file_name = self.file_names[index]
        if self.widths is not None and self.heights is not None:
            width, height = self.widths[index], self.heights[index]
        else:
            with Image.open(self.get_image_path(index)) as src:
                width, height = src.size

        if self.mask_dir is None:
            self._ensure_anns_loaded()
        anns = None if self._anns_by_image is None else self._anns_by_image.get(img_id, [])
        cat_map = self._cat_id_to_index if self._cat_id_to_index is not None else COCO_CATEGORY_ID_TO_INDEX

        try:
            return load_coco_gt_mask(
                image_id=img_id,
                height=height,
                width=width,
                anns=anns,
                cat_id_to_index=cat_map,
                size=size,
                mask_dir=self.mask_dir,
                file_name=file_name,
                ignore_crowd=self.ignore_crowd,
            )
        except Exception:
            if self._anns_by_image is None:
                self._ensure_anns_loaded()
                return load_coco_gt_mask(
                    image_id=img_id,
                    height=height,
                    width=width,
                    anns=self._anns_by_image.get(img_id, []),
                    cat_id_to_index=self._cat_id_to_index or COCO_CATEGORY_ID_TO_INDEX,
                    size=size,
                    mask_dir=None,
                    file_name=file_name,
                    ignore_crowd=self.ignore_crowd,
                )
            raise

    def __getitem__(self, index: int):
        image_path = self.get_image_path(index)
        with Image.open(image_path) as source:
            image = self.transform(source.convert("RGB")) if self.transform is not None else source.convert("RGB")
        return image, self.targets[index]


def read_split_ids(voc_root: Path, group: str, split: str) -> list[str]:
    """Read ImageSets/<group>/<split>.txt, e.g. read_split_ids(root, "Segmentation", "val")."""
    return (Path(voc_root) / "ImageSets" / group / f"{split}.txt").read_text().split()


def read_wsss_train_ids(voc_root: Path, train_split: str = "main_trainval") -> list[str]:
    """Return classifier training image IDs strictly disjoint from Segmentation/val.txt.

    - "main_trainval": ImageSets/Main/trainval.txt minus ImageSets/Segmentation/val.txt (10,418 images).
    - "seg_train": ImageSets/Segmentation/train.txt (1,464 images).
    """
    voc_root = Path(voc_root)
    val_id_set = set(read_split_ids(voc_root, "Segmentation", "val"))
    if train_split == "main_trainval":
        main_ids = read_split_ids(voc_root, "Main", "trainval")
        return [img_id for img_id in main_ids if img_id not in val_id_set]
    if train_split == "seg_train":
        seg_ids = read_split_ids(voc_root, "Segmentation", "train")
        return [img_id for img_id in seg_ids if img_id not in val_id_set]
    raise ValueError(f"train_split must be 'main_trainval' or 'seg_train', got: {train_split!r}")


def build_transforms(image_size: int, train: bool = False) -> transforms.Compose:
    """Image transforms producing [0, 1] tensors (normalization is applied later by the caller).

    train=True adds the augmentation used for classifier training; train=False is the plain
    resize used for evaluation / CAM extraction.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), value="random"),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    device: torch.device | None = None,
    seed: int | None = None,
) -> DataLoader:
    """DataLoader with the project defaults (pin_memory on CUDA, persistent workers, seeded shuffle)."""
    generator = torch.Generator().manual_seed(seed) if (shuffle and seed is not None) else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        generator=generator,
    )


def load_gt_mask(voc_root: Path, image_id: str, size: int | tuple[int, int] | None = None) -> np.ndarray:
    """Load SegmentationClass/<image_id>.png as an int64 (H, W) array.

    size: int -> resize to (size, size); (w, h) -> resize to that size; None -> keep original size.
    Resizing uses NEAREST so class ids are never interpolated.
    """
    gt_path = Path(voc_root) / "SegmentationClass" / f"{image_id}.png"
    with Image.open(gt_path) as gt_img:
        if size is not None:
            target = (size, size) if isinstance(size, int) else tuple(size)
            gt_img = gt_img.resize(target, resample=Image.NEAREST)
        return np.array(gt_img, dtype=np.int64)


def load_eval_sample(
    dataset: Dataset,
    index: int,
    image_size: int | tuple[int, int] | None = None,
) -> dict:
    """Load one image together with its segmentation ground truth for evaluation.

    Works for both `VOCClassificationDataset` and `COCOClassificationDataset`.
    - `img_tensor` always comes from `dataset[index]` (e.g. resized to 448x448) for CNN forward/CAM.
    - When `image_size is None` (default, standard benchmark protocol), `img_rgb`, `img_u8`,
      and `gt_mask` are loaded at the native image resolution `(H_orig, W_orig)` with no aspect distortion.
    - When `image_size` is provided (`int` -> `(size, size)`, or `(w, h)` tuple), `img_rgb`, `img_u8`,
      and `gt_mask` are resized to that target resolution.
    """
    img_tensor, _ = dataset[index]
    img_id = dataset.image_ids[index]
    image_path = dataset.get_image_path(index) if hasattr(dataset, "get_image_path") else (dataset.image_dir / f"{img_id}.jpg")

    if image_size is None:
        with Image.open(image_path) as source:
            img_u8 = np.array(source.convert("RGB"), dtype=np.uint8)
        img_rgb = img_u8.astype(np.float32) / 255.0
        target_wh = None
    else:
        target_wh = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        tensor_hw = (img_tensor.shape[-2], img_tensor.shape[-1])
        if tensor_hw == (target_wh[1], target_wh[0]):
            img_rgb = img_tensor.permute(1, 2, 0).numpy()
            img_u8 = to_uint8_image(img_rgb)
        else:
            with Image.open(image_path) as source:
                resized = source.convert("RGB").resize(target_wh, resample=Image.BILINEAR)
                img_u8 = np.array(resized, dtype=np.uint8)
            img_rgb = img_u8.astype(np.float32) / 255.0

    if hasattr(dataset, "load_gt_mask"):
        gt_mask = dataset.load_gt_mask(index, size=target_wh)
    else:
        gt_mask = load_gt_mask(dataset.voc_root, img_id, size=target_wh)

    if img_u8.shape[:2] != gt_mask.shape:
        raise ValueError(
            f"Image/GT shape mismatch for {img_id}: img_u8={img_u8.shape[:2]} vs gt_mask={gt_mask.shape}"
        )

    return {
        "idx": index,
        "img_id": img_id,
        "img_tensor": img_tensor,
        "img_rgb": img_rgb,
        "img_u8": img_u8,
        "gt_mask": gt_mask,
        "orig_size": gt_mask.shape,
    }


def load_eval_samples(
    dataset: Dataset,
    image_size: int | tuple[int, int] | None = None,
    indices=None,
    desc: str | None = None,
) -> list[dict]:
    """load_eval_sample for many indices (default: whole dataset at native image resolution)."""
    from tqdm.auto import tqdm

    indices = range(len(dataset)) if indices is None else indices
    iterator = tqdm(indices, desc=desc) if desc else indices
    return [load_eval_sample(dataset, i, image_size) for i in iterator]
