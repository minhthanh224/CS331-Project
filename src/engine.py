"""Training / inference loops for the multi-label classifier (shared by train.py and notebooks)."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from .constants import IMAGENET_MEAN, IMAGENET_STD

_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a (B, 3, H, W) batch of [0, 1] images."""
    return _NORMALIZE(images)


def build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float | None = None,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """AdamW with an optional separate (usually larger) learning rate for the new `fc` head."""
    if head_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=backbone_lr, weight_decay=weight_decay)
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def poly_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, power: float = 0.9):
    """Polynomial decay (1 - step / total_steps) ** power; call scheduler.step() once per iteration."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, (1.0 - min(step, total_steps) / total_steps) ** power),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    desc: str = "Training",
) -> float:
    """One training epoch. Steps `scheduler` after every iteration when given. Returns mean loss."""
    model.train()
    total_loss, total_images = 0.0, 0
    progress = tqdm(loader, desc)
    for images, targets in progress:
        inputs = normalize_batch(images).to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size
        progress.set_postfix(loss=f"{total_loss / total_images:.4f}")
    return total_loss / total_images


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    desc: str | None = None,
    leave: bool = True,
):
    """Run the classifier over a loader in eval mode.

    Returns (probs, targets, mean_loss): probs/targets are float32 NumPy arrays (N, num_classes)
    of sigmoid probabilities and ground-truth multi-hot labels; mean_loss is None unless
    `criterion` is given.
    """
    model.eval()
    probs_list, targets_list = [], []
    total_loss, total_images = 0.0, 0
    iterator = tqdm(loader, desc=desc, leave=leave) if desc else loader
    for images, targets in iterator:
        logits = model(normalize_batch(images).to(device, non_blocking=True))
        if criterion is not None:
            loss = criterion(logits, targets.to(device, non_blocking=True))
            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)
        probs_list.append(torch.sigmoid(logits).cpu())
        targets_list.append(targets.cpu())

    probs = torch.cat(probs_list, dim=0).numpy()
    targets_all = torch.cat(targets_list, dim=0).numpy()
    mean_loss = (total_loss / total_images) if criterion is not None else None
    return probs, targets_all, mean_loss
