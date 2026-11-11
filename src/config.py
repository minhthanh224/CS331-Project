from dataclasses import dataclass
from pathlib import Path

from .constants import SUPPORTED_BACKBONES

VALID_TRAIN_SPLITS = ("main_trainval", "seg_train")
VALID_LR_SCHEDULERS = ("poly", "none")


@dataclass
class TrainConfig:
    voc_root: Path
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
    train_split: str = "main_trainval"
    cls_threshold: float = 0.5
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"model_name must be one of {SUPPORTED_BACKBONES}, got: {self.model_name}")
        self.voc_root = Path(self.voc_root)

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

        if self.train_split not in VALID_TRAIN_SPLITS:
            raise ValueError(f"train_split must be one of {VALID_TRAIN_SPLITS}, got: {self.train_split}")

        if not (0.0 < self.cls_threshold < 1.0):
            raise ValueError("cls_threshold must be in (0, 1)")

        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
