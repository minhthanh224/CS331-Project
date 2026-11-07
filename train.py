from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import argparse

#Modules from src
from src.config import TrainConfig
from src.labels import build_voc_label_table
from src.dataset import VOCClassificationDataset
from src.models import build_classifier

CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

voc_root = Path.cwd() / "data" / "VOC2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-label classifier for WSSS CAM")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "resnet101"], help="Backbone model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=244, help="Input image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    return parser.parse_args()
args = parse_args()

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
print(f"Training config: model={config.model_name} | image_size={config.image_size} | epochs={config.num_epochs} | batch_size={config.batch_size} | lr={config.learning_rate}")

split_dir = config.voc_root / "ImageSets" / "Segmentation"
train_ids = (split_dir / "train.txt").read_text().split()
val_ids = (split_dir / "val.txt").read_text().split()

train_labels_df = build_voc_label_table(config.voc_root, image_ids=train_ids, classes=CLASSES)
val_labels_df = build_voc_label_table(config.voc_root, image_ids=val_ids, classes=CLASSES)

transform = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor()
    ]
)

train_dataset = VOCClassificationDataset(voc_root=voc_root, labels_df=train_labels_df, classes=CLASSES, transform=transform)
val_dataset = VOCClassificationDataset(voc_root=voc_root, labels_df=val_labels_df, classes=CLASSES, transform=transform)

model = build_classifier(model_name=config.model_name, num_classes=20)

train_loader = DataLoader(
    train_dataset,
    batch_size = config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    generator=torch.Generator().manual_seed(config.seed)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
)

#Normalizer for images
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

model.to(device)

#Run epoch function with train and validation mode
def run_epoch(is_training: bool, loader: DataLoader):
    model.train(is_training)

    total_loss = 0
    total_images = 0

    #progress bar UI using tqdm
    progress = tqdm(loader, "Training" if is_training else "Validating")

    with torch.set_grad_enabled(is_training):
        for images, targets in progress:
            model_input = normalize(images).to(device)
            targets = targets.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(model_input)
            loss = criterion(logits, targets)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size
    return total_loss / total_images

best_loss = float("inf")
checkpoint_dir = Path("checkpoints") / config.model_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(1, config.num_epochs + 1):
    print(f"Epoch {epoch}/{config.num_epochs}")
    train_loss = run_epoch(True, train_loader)
    val_loss = run_epoch(False, val_loader)
    print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        print(f"Saved best model | Val loss: {best_loss:.4f}")

torch.save(model.state_dict(), checkpoint_dir / "last.pt")
print("Training completed !")
