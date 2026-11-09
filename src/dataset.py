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


def load_eval_sample(dataset: VOCClassificationDataset, index: int, image_size: int | None = None) -> dict:
    """Load one image together with its segmentation ground truth for evaluation.

    Returns a dict with: idx, img_id, img_tensor (3,H,W in [0,1]), img_rgb (H,W,3 float),
    img_u8 (H,W,3 uint8) and gt_mask (H,W int64, resized to image_size when given).
    """
    img_tensor, _ = dataset[index]
    img_rgb = img_tensor.permute(1, 2, 0).numpy()
    img_id = dataset.image_ids[index]
    return {
        "idx": index,
        "img_id": img_id,
        "img_tensor": img_tensor,
        "img_rgb": img_rgb,
        "img_u8": to_uint8_image(img_rgb),
        "gt_mask": load_gt_mask(dataset.voc_root, img_id, image_size),
    }


def load_eval_samples(dataset: VOCClassificationDataset, image_size: int | None = None, indices=None, desc: str | None = None) -> list[dict]:
    """load_eval_sample for many indices (default: the whole dataset), with an optional tqdm bar."""
    from tqdm.auto import tqdm

    indices = range(len(dataset)) if indices is None else indices
    iterator = tqdm(indices, desc=desc) if desc else indices
    return [load_eval_sample(dataset, i, image_size) for i in iterator]
