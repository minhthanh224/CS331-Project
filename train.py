import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from src.config import VALID_LR_SCHEDULERS, TrainConfig
from src.constants import SUPPORTED_BACKBONES, get_dataset_classes, normalize_dataset_name
from src.dataset import (
    COCOClassificationDataset,
    VOCClassificationDataset,
    build_transforms,
    make_loader,
    read_split_ids,
    read_wsss_train_ids,
    resolve_coco_paths,
)
from src.engine import build_optimizer, poly_scheduler, predict_probs, train_one_epoch
from src.labels import build_coco_label_table, build_voc_label_table
from src.metrics import compute_classification_metrics
from src.models import build_classifier
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label classifier for WSSS CAM / LayerCAM")
    parser.add_argument(
        "--dataset",
        type=str,
        default="voc2012",
        choices=["voc2012", "voc", "coco", "coco2014", "coco2017"],
        help="Dataset name: 'voc2012' (20 classes) or 'coco'/'coco2014'/'coco2017' (80 classes)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to dataset root (defaults to data/VOC2012 for voc2012, data/COCO for coco)",
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=None,
        help="Deprecated alias for --data-root when using VOC2012",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=list(SUPPORTED_BACKBONES),
        help="Backbone model name",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=None,
        help="Training split (VOC: 'main_trainval' or 'seg_train'; COCO: 'train2017', 'train2014', or 'train')",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default=None,
        help="Validation split (VOC: 'val'; COCO: 'val2017', 'val2014', or 'val')",
    )
    parser.add_argument(
        "--drop-empty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop images with 0 object annotations from the COCO training split (default: False)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Backbone learning rate")
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
        help="Classification head learning rate (set 0 to share --lr with backbone)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="poly",
        choices=list(VALID_LR_SCHEDULERS),
        help="Learning rate schedule ('poly' per-iteration decay or 'none')",
    )
    parser.add_argument("--poly-power", type=float, default=0.9, help="Exponent for polynomial LR scheduler")
    parser.add_argument("--image-size", type=int, default=448, help="Input image size for classifier")
    parser.add_argument(
        "--layer4-stride1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set the last-stage stride to (1, 1) for ~2x higher CAM resolution: ResNet layer4[0] / ConvNeXt last downsample (default: True)",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA automatic mixed precision (FP16) during training/eval (default: False)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume training from last.pt if present in checkpoint directory (default: False)",
    )
    parser.add_argument("--cls-threshold", type=float, default=0.5, help="Validation F1 probability threshold")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Base directory to save model checkpoints and training history",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    return parser.parse_args()


def _serialize_train_config(config: TrainConfig) -> dict:
    raw = asdict(config)
    raw["voc_root"] = str(config.voc_root)
    raw["data_root"] = str(config.data_root)
    return raw


def _write_history_csv(history: list[dict], csv_path: Path) -> None:
    if not history:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _build_datasets(config: TrainConfig, classes: tuple[str, ...]):
    train_transform = build_transforms(config.image_size, train=True)
    eval_transform = build_transforms(config.image_size, train=False)

    if config.dataset == "voc2012":
        train_ids = read_wsss_train_ids(config.data_root, train_split=config.train_split)
        val_ids = read_split_ids(config.data_root, "Segmentation", "val")
        train_labels_df = build_voc_label_table(config.data_root, image_ids=train_ids, classes=classes)
        val_labels_df = build_voc_label_table(config.data_root, image_ids=val_ids, classes=classes)
        train_dataset = VOCClassificationDataset(config.data_root, train_labels_df, classes, train_transform)
        val_dataset = VOCClassificationDataset(config.data_root, val_labels_df, classes, eval_transform)
        return train_dataset, val_dataset

    train_img_dir, train_ann_file = resolve_coco_paths(config.data_root, split=config.train_split)
    val_img_dir, val_ann_file = resolve_coco_paths(config.data_root, split=config.val_split)
    train_labels_df = build_coco_label_table(train_ann_file, classes=classes, drop_empty=config.drop_empty)
    val_labels_df = build_coco_label_table(val_ann_file, classes=classes, drop_empty=False)
    train_dataset = COCOClassificationDataset(
        config.data_root,
        train_labels_df,
        classes=classes,
        transform=train_transform,
        split=config.train_split,
        image_dir=train_img_dir,
        ann_file=train_ann_file,
    )
    val_dataset = COCOClassificationDataset(
        config.data_root,
        val_labels_df,
        classes=classes,
        transform=eval_transform,
        split=config.val_split,
        image_dir=val_img_dir,
        ann_file=val_ann_file,
    )
    return train_dataset, val_dataset


def main() -> None:
    args = parse_args()
    device = get_device()
    head_lr = None if args.head_lr == 0 else args.head_lr
    norm_dataset = normalize_dataset_name(args.dataset)
    default_root = Path.cwd() / "data" / ("VOC2012" if norm_dataset == "voc2012" else "COCO")
    data_root = args.data_root if args.data_root is not None else (args.voc_root or default_root)

    config = TrainConfig(
        dataset=args.dataset,
        data_root=data_root,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        head_lr=head_lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        poly_power=args.poly_power,
        layer4_stride1=args.layer4_stride1,
        train_split=args.train_split,
        val_split=args.val_split,
        drop_empty=args.drop_empty,
        amp=args.amp,
        cls_threshold=args.cls_threshold,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    set_seed(config.seed)
    classes = get_dataset_classes(config.dataset)

    train_dataset, val_dataset = _build_datasets(config, classes)

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

    model = build_classifier(
        model_name=config.model_name,
        num_classes=len(classes),
        layer4_stride1=config.layer4_stride1,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(
        model,
        backbone_lr=config.learning_rate,
        head_lr=config.head_lr,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        poly_scheduler(optimizer, total_steps=config.num_epochs * len(train_loader), power=config.poly_power)
        if config.lr_scheduler == "poly"
        else None
    )
    scaler = torch.amp.GradScaler("cuda") if (config.amp and device.type == "cuda") else None

    print(
        f"Training config: dataset={config.dataset} ({len(classes)} classes) | model={config.model_name} | "
        f"split={config.train_split} ({len(train_dataset)} train / {len(val_dataset)} val) | "
        f"image_size={config.image_size} | layer4_stride1={config.layer4_stride1} | "
        f"epochs={config.num_epochs} | batch_size={config.batch_size} | lr={config.learning_rate} | "
        f"head_lr={config.head_lr} | scheduler={config.lr_scheduler} | amp={config.amp} | device={device}"
    )

    checkpoint_base = Path(args.checkpoint_dir)
    if config.dataset != "voc2012":
        checkpoint_base = checkpoint_base / config.dataset
    checkpoint_dir = checkpoint_base / config.model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dict = _serialize_train_config(config)

    best_map = -1.0
    best_epoch = 0
    history: list[dict] = []
    start_epoch = 1

    last_ckpt_path = checkpoint_dir / "last.pt"
    if args.resume and last_ckpt_path.is_file():
        resume_state = torch.load(last_ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_state["model_state_dict"])
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if scheduler is not None and resume_state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if scaler is not None and resume_state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        history = list(resume_state.get("history", []))
        best_map = float(resume_state.get("best_val_mAP", -1.0))
        best_epoch = int(resume_state.get("best_epoch", 0))
        start_epoch = int(resume_state.get("epoch_in_run", 0)) + 1
        print(f"Resumed from {last_ckpt_path} (epoch {start_epoch - 1}/{config.num_epochs}, best Val mAP={best_map:.4f})")

    for epoch in range(start_epoch, config.num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
            desc=f"[{config.model_name}] Epoch {epoch:02d}/{config.num_epochs:02d} [Train]",
            use_amp=config.amp,
            scaler=scaler,
        )
        val_probs, val_targets, val_loss = predict_probs(
            model,
            val_loader,
            device,
            criterion=criterion,
            desc=f"[{config.model_name}] Epoch {epoch:02d}/{config.num_epochs:02d} [Val]",
            leave=False,
            use_amp=config.amp,
        )
        val_metrics = compute_classification_metrics(val_targets, val_probs, threshold=config.cls_threshold)
        val_map = val_metrics["mAP"]
        val_macro_f1 = val_metrics["macro_f1"]

        row = {
            "dataset": config.dataset,
            "backbone": config.model_name,
            "epoch": epoch,
            "backbone_lr": current_lr,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_mAP": round(val_map, 6),
            "val_macro_f1": round(val_macro_f1, 6),
        }
        history.append(row)

        improved = val_map > best_map
        if improved:
            best_map = val_map
            best_epoch = epoch

        best_payload = {
            "dataset": config.dataset,
            "backbone": config.model_name,
            "epoch_in_run": epoch,
            "best_epoch": best_epoch,
            "num_classes": len(classes),
            "classes": list(classes),
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "val_mAP": val_map,
            "best_val_mAP": best_map,
            "history": list(history),
            "config": config_dict,
        }
        last_payload = {
            **best_payload,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        }
        torch.save(last_payload, last_ckpt_path)
        if improved:
            torch.save(best_payload, checkpoint_dir / "best.pt")
            torch.save(model.state_dict(), checkpoint_dir / "best_weights_only.pt")

        (checkpoint_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        _write_history_csv(history, checkpoint_dir / "training_history.csv")

        star = " [BEST]" if improved else ""
        print(
            f"[{config.model_name}] Epoch {epoch:02d}/{config.num_epochs:02d} | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mAP: {val_map:.4f} | Macro F1: {val_macro_f1:.4f}{star}"
        )

    print(f"Training completed! Best Val mAP: {best_map:.4f} (Epoch {best_epoch}) -> {checkpoint_dir}")


if __name__ == "__main__":
    main()
