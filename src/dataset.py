from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class VOCClassificationDataset(Dataset):
    def __init__(self, voc_root:Path, labels_df:pd.DataFrame, classes: tuple[str, ...], transform,) -> None:
        self.image_dir = voc_root / "JPEGImages"
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
