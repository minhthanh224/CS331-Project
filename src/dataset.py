from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from .utils import to_uint8_image


class VOCClassificationDataset(Dataset):
    def __init__(self, voc_root:Path, labels_df:pd.DataFrame, classes: tuple[str, ...], transform,) -> None:
        self.voc_root = Path(voc_root)
        self.image_dir = self.voc_root / "JPEGImages"
        self.image_ids = labels_df["image_id"].tolist()
        self.targets = torch.tensor(labels_df[list(classes)].to_numpy(dtype="float32"))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_path = self.image_dir / f"{self.image_ids[index]}.jpg"
        with Image.open(image_path) as source:
            image = self.transform(source.convert("RGB"))
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
    dataset: VOCClassificationDataset,
    index: int,
    image_size: int | tuple[int, int] | None = None,
) -> dict:
    """Load one image together with its segmentation ground truth for evaluation.

    - `img_tensor` always comes from `dataset[index]` (e.g. resized to 448x448) for CNN forward/CAM.
    - When `image_size is None` (default, standard Pascal VOC benchmark protocol), `img_rgb`, `img_u8`,
      and `gt_mask` are loaded at the native image resolution `(H_orig, W_orig)` with no aspect distortion.
    - When `image_size` is provided (`int` -> `(size, size)`, or `(w, h)` tuple), `img_rgb`, `img_u8`,
      and `gt_mask` are resized to that target resolution.
    """
    img_tensor, _ = dataset[index]
    img_id = dataset.image_ids[index]
    image_path = dataset.image_dir / f"{img_id}.jpg"

    if image_size is None:
        with Image.open(image_path) as source:
            img_u8 = np.array(source.convert("RGB"), dtype=np.uint8)
        img_rgb = img_u8.astype(np.float32) / 255.0
        gt_mask = load_gt_mask(dataset.voc_root, img_id, size=None)
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
    dataset: VOCClassificationDataset,
    image_size: int | tuple[int, int] | None = None,
    indices=None,
    desc: str | None = None,
) -> list[dict]:
    """load_eval_sample for many indices (default: whole dataset at native image resolution)."""
    from tqdm.auto import tqdm

    indices = range(len(dataset)) if indices is None else indices
    iterator = tqdm(indices, desc=desc) if desc else indices
    return [load_eval_sample(dataset, i, image_size) for i in iterator]
