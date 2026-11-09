import argparse
from pathlib import Path

import torch
from torch import nn

from src.config import TrainConfig
from src.constants import SUPPORTED_BACKBONES, VOC_CLASSES
from src.dataset import (
    VOCClassificationDataset,
    build_transforms,
    make_loader,
    read_split_ids,
)
from src.engine import predict_probs, train_one_epoch
from src.labels import build_voc_label_table
from src.models import build_classifier
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label classifier for WSSS CAM")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=list(SUPPORTED_BACKBONES),
        help="Backbone model name",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=244, help="Input image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    voc_root = Path.cwd() / "data" / "VOC2012"
    device = get_device()

    config = TrainConfig(
        voc_root=voc_root,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    set_seed(config.seed)
    print(
        f"Training config: model={config.model_name} | image_size={config.image_size} | "
        f"epochs={config.num_epochs} | batch_size={config.batch_size} | lr={config.learning_rate}"
    )

    train_ids = read_split_ids(config.voc_root, "Segmentation", "train")
    val_ids = read_split_ids(config.voc_root, "Segmentation", "val")

    train_labels_df = build_voc_label_table(config.voc_root, image_ids=train_ids, classes=VOC_CLASSES)
    val_labels_df = build_voc_label_table(config.voc_root, image_ids=val_ids, classes=VOC_CLASSES)

    transform = build_transforms(config.image_size, train=False)
    train_dataset = VOCClassificationDataset(config.voc_root, train_labels_df, VOC_CLASSES, transform)
    val_dataset = VOCClassificationDataset(config.voc_root, val_labels_df, VOC_CLASSES, transform)

    train_loader = make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        device=device,
        seed=config.seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device,
    )

    model = build_classifier(model_name=config.model_name, num_classes=len(VOC_CLASSES)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    checkpoint_dir = Path("checkpoints") / config.model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.num_epochs + 1):
        print(f"Epoch {epoch}/{config.num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, desc="Training")
        _, _, val_loss = predict_probs(model, val_loader, device, criterion=criterion, desc="Validating")
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"Saved best model | Val loss: {best_loss:.4f}")

    torch.save(model.state_dict(), checkpoint_dir / "last.pt")
    print("Training completed !")


if __name__ == "__main__":
    main()
