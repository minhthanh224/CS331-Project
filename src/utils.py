"""Small helpers shared by notebooks and scripts (seed, device, paths, image conversion)."""
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(require_gpu: bool = False) -> torch.device:
    """Return the CUDA device when available, otherwise CPU.

    With require_gpu=True a missing GPU raises instead of silently falling back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError("GPU is required but not available (Colab: Runtime -> Change runtime type -> GPU).")
    return torch.device("cpu")


def in_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from `start` (default: cwd) to the directory that contains the `src` package."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src" / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find the project root (a folder containing src/) above {current}")


def add_project_root_to_path(start: Path | None = None) -> Path:
    """Make `import src...` work from any notebook folder. Returns the project root."""
    root = find_project_root(start)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def to_uint8_image(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image in [0, 1] (float) or [0, 255] to uint8 [0, 255]."""
    if image_rgb.dtype == np.uint8:
        return image_rgb
    if image_rgb.max() <= 1.0:
        return (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return image_rgb.astype(np.uint8)
