"""Plot helpers for the image / ground-truth / CAM / mask grids repeated across notebooks."""
from __future__ import annotations

import numpy as np

# A grid cell is a dict: {"kind": ..., "data": ..., "title": ..., ["image": ..., "alpha": ...]}
#   kind "image"   -> data is an RGB image
#   kind "mask"    -> data is an integer label map (VOC palette-like colormap, 0..20)
#   kind "heatmap" -> data is a CAM in [0, 1]
#   kind "overlay" -> data is a CAM drawn over cell["image"] with transparency cell.get("alpha", 0.5)


def _build_segmentation_palette(num_colors: int = 256) -> np.ndarray:
    """Deterministic high-contrast RGB palette (0 = black background, 255 = neutral gray ignore)."""
    palette = np.zeros((max(num_colors, 256), 3), dtype=np.uint8)
    for idx in range(1, 255):
        r = g = b = 0
        c = idx
        for bit in range(8):
            r |= ((c >> 0) & 1) << (7 - bit)
            g |= ((c >> 1) & 1) << (7 - bit)
            b |= ((c >> 2) & 1) << (7 - bit)
            c >>= 3
        palette[idx] = (r, g, b)
    palette[255] = (128, 128, 128)
    return palette


_SEG_PALETTE = _build_segmentation_palette()


def image_cell(image: np.ndarray, title: str = "") -> dict:
    return {"kind": "image", "data": image, "title": title}


def mask_cell(mask: np.ndarray, title: str = "", num_classes: int | None = None) -> dict:
    return {"kind": "mask", "data": mask, "title": title, "num_classes": num_classes}


def heatmap_cell(cam: np.ndarray, title: str = "") -> dict:
    return {"kind": "heatmap", "data": cam, "title": title}


def overlay_cell(image: np.ndarray, cam: np.ndarray, title: str = "", alpha: float = 0.5) -> dict:
    return {"kind": "overlay", "data": cam, "image": image, "title": title, "alpha": alpha}


def draw_cell(ax, cell: dict) -> None:
    kind = cell["kind"]
    if kind == "image":
        ax.imshow(cell["data"])
    elif kind == "mask":
        mask = np.asarray(cell["data"])
        num_classes = cell.get("num_classes")
        valid = mask[mask != 255]
        max_val = int(valid.max()) if valid.size > 0 else 0
        if (num_classes is not None and num_classes > 21) or max_val > 20:
            ax.imshow(_SEG_PALETTE[np.clip(mask, 0, 255)])
        else:
            ax.imshow(mask, cmap="tab20", vmin=0, vmax=20)
    elif kind == "heatmap":
        ax.imshow(cell["data"], cmap="jet", vmin=0, vmax=1)
    elif kind == "overlay":
        ax.imshow(cell["image"])
        ax.imshow(cell["data"], cmap="jet", alpha=cell.get("alpha", 0.5), vmin=0, vmax=1)
    else:
        raise ValueError(f"Unknown cell kind: {kind}")
    ax.set_title(cell.get("title", ""))
    ax.axis("off")


def plot_grid(rows: list[list[dict]], cell_size: float = 4.0, save_path=None, show: bool = True):
    """Draw a grid (one list of cells per row) and optionally save it. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell_size * n_cols, cell_size * n_rows), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            if c < len(row):
                draw_cell(axes[r, c], row[c])
            else:
                axes[r, c].axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig
