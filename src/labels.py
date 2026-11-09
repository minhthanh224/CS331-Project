from pathlib import Path
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from .constants import VOC_CLASSES

def encode_multihot(labels: set[str], class_to_index: dict[str, int]) -> np.ndarray:
    ids = np.zeros(len(class_to_index), dtype=np.float32)
    for label in labels:
        if label in class_to_index:
            ids[class_to_index[label]] = 1
    return ids

def build_class_to_index(classes: tuple [str, ...] = VOC_CLASSES) -> dict[str, int]:
    return dict(zip(classes, range(len(classes))))

def read_voc_labels(xml_path: Path) -> set[str]:
    annotation = ET.parse(xml_path).getroot()
    names = {obj.findtext("name") for obj in annotation.findall("object")}
    return names

def build_voc_label_table(
    voc_root: Path,
    image_ids: list[str],
    classes: tuple[str, ...] = VOC_CLASSES,
) -> pd.DataFrame:
    rows = []
    for image_id in image_ids:
        names = read_voc_labels(voc_root / "Annotations" / f"{image_id}.xml")
        row = {"image_id": image_id}
        row.update({name: int(name in names) for name in classes})
        rows.append(row)

    return pd.DataFrame(rows, columns=["image_id", *classes])


def build_balanced_tune_table(
    df_seg_train_labels: pd.DataFrame,
    train_ids: list[str] | set[str],
    classes: tuple[str, ...] = VOC_CLASSES,
    max_images: int = 260,
    max_person_only: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a class-balanced tuning subset from Segmentation/train.txt, prioritizing images not in `train_ids`
    and capping 'person-only' images so person does not dominate hyperparameter tuning.
    """
    seg_train_ids = df_seg_train_labels["image_id"].tolist()
    non_overlap_ids = sorted(set(seg_train_ids) - set(train_ids))
    df_non_overlap = df_seg_train_labels[df_seg_train_labels["image_id"].isin(non_overlap_ids)].copy()

    other_cols = [c for c in classes if c != "person"]
    only_person = (df_non_overlap["person"] == 1) & (df_non_overlap[other_cols].sum(axis=1) == 0)

    df_tune = pd.concat(
        [
            df_non_overlap[~only_person],
            df_non_overlap[only_person].sample(n=min(max_person_only, int(only_person.sum())), random_state=seed),
        ],
        ignore_index=True,
    )

    if len(df_tune) < max_images:
        rem_df = df_seg_train_labels[~df_seg_train_labels["image_id"].isin(df_tune["image_id"])]
        rem_non_person = rem_df[(rem_df["person"] == 0) | (rem_df[other_cols].sum(axis=1) > 0)]
        extra_n = min(max_images - len(df_tune), len(rem_non_person))
        df_tune = pd.concat(
            [df_tune, rem_non_person.sample(n=extra_n, random_state=seed)],
            ignore_index=True,
        )

    return df_tune.sort_values("image_id").reset_index(drop=True)

