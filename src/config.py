from dataclasses import dataclass
from pathlib import Path

from .constants import SUPPORTED_BACKBONES, normalize_dataset_name

VALID_DATASETS = ("voc2012", "coco")
VALID_TRAIN_SPLITS = ("main_trainval", "seg_train")
VALID_COCO_SPLITS = ("train2017", "val2017", "train2014", "val2014", "train", "val")
VALID_LR_SCHEDULERS = ("poly", "none")


@dataclass
class TrainConfig:
    voc_root: Path | None = None
    model_name: str = "resnet50"
    image_size: int = 448
    batch_size: int = 16
    num_epochs: int = 15
    learning_rate: float = 1e-4
    head_lr: float | None = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "poly"
    poly_power: float = 0.9
    layer4_stride1: bool = True
    train_split: str | None = None
    cls_threshold: float = 0.5
    seed: int = 42
    num_workers: int = 0
    dataset: str = "voc2012"
    data_root: Path | None = None
    val_split: str | None = None
    drop_empty: bool = False
    amp: bool = False

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"model_name must be one of {SUPPORTED_BACKBONES}, got: {self.model_name}")

        raw_dataset = self.dataset.strip().lower()
        self.dataset = normalize_dataset_name(self.dataset)

        root = self.data_root if self.data_root is not None else self.voc_root
        if root is None:
            raise ValueError("Either data_root or voc_root must be provided.")
        self.data_root = Path(root)
        self.voc_root = self.data_root

        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")

        if self.batch_size <= 0 or self.num_epochs <= 0:
            raise ValueError("batch_size and num_epochs must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.head_lr is not None and self.head_lr <= 0:
            raise ValueError("head_lr must be positive when specified")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")

        if self.lr_scheduler not in VALID_LR_SCHEDULERS:
            raise ValueError(f"lr_scheduler must be one of {VALID_LR_SCHEDULERS}, got: {self.lr_scheduler}")

        if self.poly_power <= 0:
            raise ValueError("poly_power must be positive")

        if self.dataset == "voc2012":
            if self.train_split is None:
                self.train_split = "main_trainval"
            if self.train_split not in VALID_TRAIN_SPLITS:
                raise ValueError(f"train_split must be one of {VALID_TRAIN_SPLITS}, got: {self.train_split}")
            if self.val_split is None:
                self.val_split = "val"
            elif self.val_split not in ("val", "seg_val"):
                raise ValueError(f"val_split for voc2012 must be 'val' or 'seg_val', got: {self.val_split}")
        else:
            default_year = "2014" if "2014" in raw_dataset else "2017"
            if self.train_split is None:
                self.train_split = f"train{default_year}"
            if self.val_split is None:
                self.val_split = (
                    "val2014" if "2014" in self.train_split else (
                        "val" if self.train_split == "train" else f"val{default_year}"
                    )
                )
            if self.train_split not in VALID_COCO_SPLITS:
                raise ValueError(f"COCO train_split must be one of {VALID_COCO_SPLITS}, got: {self.train_split}")
            if self.val_split not in VALID_COCO_SPLITS:
                raise ValueError(f"COCO val_split must be one of {VALID_COCO_SPLITS}, got: {self.val_split}")

        if not (0.0 < self.cls_threshold < 1.0):
            raise ValueError("cls_threshold must be in (0, 1)")

        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
