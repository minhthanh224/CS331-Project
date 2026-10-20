from pathlib import Path
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

def encode_multihot(labels: set[str], class_to_index: dict[str, int]) -> np.ndarray:
    ids = np.zeros(len(class_to_index), dtype=np.float32)
    for label in labels:
        if label in class_to_index:
            ids[class_to_index[label]] = 1
    return ids

def build_class_to_index(classes: tuple [str, ...]) -> dict[str, int]:
    return dict(zip(classes, range(len(classes))))

def read_voc_labels(xml_path: Path) -> set[str]:
    annotation = ET.parse(xml_path).getroot()
    names = {obj.findtext("name") for obj in annotation.findall("object")}
    return names

def build_voc_label_table(voc_root: Path, image_ids: list[str], classes: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for image_id in image_ids:
        names = read_voc_labels(voc_root / "Annotations" / f"{image_id}.xml")
        row = {"image_id": image_id}
        row.update({name: int(name in names) for name in classes})
        rows.append(row)

    return pd.DataFrame(rows, columns=["image_id", *classes])
