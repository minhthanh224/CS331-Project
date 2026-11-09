from dataclasses import dataclass
from pathlib import Path

from .constants import SUPPORTED_BACKBONES

@dataclass
class TrainConfig:
    voc_root : Path
    model_name: str = "resnet18"
    image_size: int = 244
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-4
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_BACKBONES:
            raise ValueError("model_name must be resnet18, resnet50, or resnet101")
        self.voc_root = Path(self.voc_root)

        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")

        if self.batch_size <= 0 or self.num_epochs <= 0:
            raise ValueError("batch_size and num_epochs must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
