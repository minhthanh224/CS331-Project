import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .constants import COCO_CATEGORY_ID_TO_INDEX, COCO_CLASSES, VOC_CLASSES


def encode_multihot(labels: set[str], class_to_index: dict[str, int]) -> np.ndarray:
    ids = np.zeros(len(class_to_index), dtype=np.float32)
    for label in labels:
        if label in class_to_index:
            ids[class_to_index[label]] = 1
    return ids


def build_class_to_index(classes: tuple[str, ...] = VOC_CLASSES) -> dict[str, int]:
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

    if "person" in classes:
        other_cols = [c for c in classes if c != "person"]
        only_person = (df_non_overlap["person"] == 1) & (df_non_overlap[other_cols].sum(axis=1) == 0)
        df_tune = pd.concat(
            [
                df_non_overlap[~only_person],
                df_non_overlap[only_person].sample(n=min(max_person_only, int(only_person.sum())), random_state=seed),
            ],
            ignore_index=True,
        )
    else:
        df_tune = df_non_overlap

    if len(df_tune) < max_images:
        rem_df = df_seg_train_labels[~df_seg_train_labels["image_id"].isin(df_tune["image_id"])]
        if "person" in classes:
            other_cols = [c for c in classes if c != "person"]
            rem_non_person = rem_df[(rem_df["person"] == 0) | (rem_df[other_cols].sum(axis=1) > 0)]
        else:
            rem_non_person = rem_df
        extra_n = min(max_images - len(df_tune), len(rem_non_person))
        df_tune = pd.concat(
            [df_tune, rem_non_person.sample(n=extra_n, random_state=seed)],
            ignore_index=True,
        )

    return df_tune.sort_values("image_id").reset_index(drop=True)


def load_coco_json(ann_file: Path | str) -> dict:
    """Load an MS COCO instance annotations JSON file."""
    ann_path = Path(ann_file)
    if not ann_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_path}")
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_coco_cat_id_to_index(
    categories: list[dict] | None = None,
    classes: tuple[str, ...] = COCO_CLASSES,
) -> dict[int, int]:
    """Map COCO JSON `category_id` values to contiguous 0-based indices for `classes`."""
    if tuple(classes) == COCO_CLASSES:
        mapping = dict(COCO_CATEGORY_ID_TO_INDEX)
        if categories:
            name_to_idx = build_class_to_index(classes)
            for cat in categories:
                name = cat.get("name")
                if name in name_to_idx:
                    mapping[int(cat["id"])] = name_to_idx[name]
        return mapping

    if not categories:
        raise ValueError("`categories` from COCO JSON is required when using custom `classes`.")
    name_to_idx = build_class_to_index(classes)
    return {
        int(cat["id"]): name_to_idx[cat["name"]]
        for cat in categories
        if cat.get("name") in name_to_idx
    }


def build_coco_label_table(
    ann_file: Path | str,
    classes: tuple[str, ...] = COCO_CLASSES,
    drop_empty: bool = False,
    coco_data: dict | None = None,
) -> pd.DataFrame:
    """Build a multi-hot classification DataFrame from a COCO `instances_*.json` file.

    Returns a DataFrame sorted by `image_id` with columns:
    `["image_id", "file_name", "width", "height", *classes]`.
    """
    data = coco_data if coco_data is not None else load_coco_json(ann_file)
    images = sorted(data["images"], key=lambda img: int(img["id"]))
    cat_to_idx = build_coco_cat_id_to_index(data.get("categories"), classes=classes)

    image_ids = [int(img["id"]) for img in images]
    id_to_row = {img_id: idx for idx, img_id in enumerate(image_ids)}
    matrix = np.zeros((len(images), len(classes)), dtype=np.uint8)

    for ann in data.get("annotations", ()):
        row_idx = id_to_row.get(int(ann["image_id"]))
        cls_idx = cat_to_idx.get(int(ann["category_id"]))
        if row_idx is not None and cls_idx is not None:
            matrix[row_idx, cls_idx] = 1

    meta_df = pd.DataFrame({
        "image_id": image_ids,
        "file_name": [img["file_name"] for img in images],
        "width": [int(img["width"]) for img in images],
        "height": [int(img["height"]) for img in images],
    })
    label_df = pd.DataFrame(matrix, columns=list(classes))
    df = pd.concat([meta_df, label_df], axis=1)

    if drop_empty:
        has_fg = matrix.sum(axis=1) > 0
        df = df.loc[has_fg].reset_index(drop=True)
    return df


def build_coco_balanced_tune_table(
    df_labels: pd.DataFrame,
    exclude_ids: list | set = (),
    classes: tuple[str, ...] = COCO_CLASSES,
    max_images: int = 240,
    tune_fraction: float | None = None,
    min_per_class: int = 3,
    max_person_only: int | None = 6,
    max_class_fraction: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample a class-balanced COCO tuning subset covering rare classes without starving Train.

    Excludes `exclude_ids` and empty (0-label) images. Size is determined by `tune_fraction`
    (when set, e.g. 0.05 for 5% of non-empty pool) or `max_images`. Phase 1 covers rare classes up
    to `min_per_class`; Phase 2 iteratively picks an unused non-person-only image for the currently
    least-covered class (capped at `max_class_fraction` so Train retains >= 75% of every class).
    """
    pool = df_labels[~df_labels["image_id"].isin(set(exclude_ids))].copy()
    class_cols = list(classes)
    class_mat = pool[class_cols].to_numpy(dtype=np.int64)
    non_empty_mask = class_mat.sum(axis=1) > 0
    pool = pool.loc[non_empty_mask].reset_index(drop=True)
    if len(pool) == 0:
        return pool.iloc[:0].copy()

    if tune_fraction is not None:
        if not (0.0 < tune_fraction < 1.0):
            raise ValueError(f"tune_fraction must be in (0, 1), got: {tune_fraction}")
        target_images = min(len(pool), max(1, int(round(len(pool) * tune_fraction))))
    else:
        target_images = min(len(pool), int(max_images))

    if target_images <= 0:
        return pool.iloc[:0].copy()

    class_mat = pool[class_cols].to_numpy(dtype=np.int64)
    num_classes = len(class_cols)
    rng = np.random.default_rng(seed)

    if "person" in class_cols:
        person_idx = class_cols.index("person")
        only_person = (class_mat[:, person_idx] == 1) & (class_mat.sum(axis=1) == 1)
    else:
        only_person = np.zeros(len(pool), dtype=bool)

    person_cap = (
        max(6, int(round(target_images * 0.02)))
        if max_person_only is None
        else max(0, int(max_person_only))
    )

    totals = class_mat.sum(axis=0)
    max_safe = np.where(totals >= 2, totals - 1, totals)
    if 0.0 < max_class_fraction < 1.0:
        frac_cap = np.maximum(min_per_class, np.floor(totals * max_class_fraction).astype(np.int64))
        class_cap = np.minimum(frac_cap, max_safe)
    else:
        class_cap = max_safe

    cls_cands: list[np.ndarray] = []
    for c in range(num_classes):
        cands = np.where((class_mat[:, c] == 1) & (~only_person))[0]
        if len(cands) == 0:
            cands = np.where(class_mat[:, c] == 1)[0]
        rng.shuffle(cands)
        cls_cands.append(cands)

    used = np.zeros(len(pool), dtype=bool)
    selected_pos: list[int] = []
    class_counts = np.zeros(num_classes, dtype=np.int64)
    ptrs = np.zeros(num_classes, dtype=np.int64)

    def _next_unused(cls_idx: int) -> int | None:
        cands = cls_cands[cls_idx]
        p = int(ptrs[cls_idx])
        while p < len(cands) and used[cands[p]]:
            p += 1
        if p >= len(cands):
            ptrs[cls_idx] = len(cands)
            return None
        ptrs[cls_idx] = p + 1
        return int(cands[p])

    def _select(pos_i: int) -> None:
        used[pos_i] = True
        selected_pos.append(pos_i)
        class_counts[:] += class_mat[pos_i]

    # Phase 1: rarest classes first up to min_per_class.
    class_order = [int(c) for c in np.argsort(totals, kind="stable") if totals[c] > 0]
    for cls_idx in class_order:
        if len(selected_pos) >= target_images:
            break
        phase1_target = min(int(min_per_class), int(class_cap[cls_idx]))
        while class_counts[cls_idx] < phase1_target and len(selected_pos) < target_images:
            pos_i = _next_unused(cls_idx)
            if pos_i is None:
                break
            _select(pos_i)

    # Phase 2: balanced round-robin targeting the currently least-covered class under class_cap.
    exhausted = np.array([int(ptrs[c]) >= len(cls_cands[c]) or totals[c] == 0 for c in range(num_classes)], dtype=bool)
    while len(selected_pos) < target_images:
        active = [
            c for c in range(num_classes)
            if (not exhausted[c]) and class_counts[c] < class_cap[c]
        ]
        if not active:
            break
        cls_idx = min(active, key=lambda c: (int(class_counts[c]), int(totals[c]), c))
        pos_i = _next_unused(cls_idx)
        if pos_i is None:
            exhausted[cls_idx] = True
            continue
        _select(pos_i)

    # Phase 3: bounded person-only allowance.
    if len(selected_pos) < target_images and person_cap > 0 and only_person.any():
        person_candidates = np.where(only_person & (~used))[0]
        rng.shuffle(person_candidates)
        added_person = 0
        for pos in person_candidates:
            _select(int(pos))
            added_person += 1
            if added_person >= person_cap or len(selected_pos) >= target_images:
                break

    # Phase 4: safety fill if target_images exceeds Phase 1-3 caps (keeps classes least-skewed).
    while len(selected_pos) < target_images:
        active = [
            c for c in range(num_classes)
            if (not exhausted[c]) and class_counts[c] < max_safe[c]
        ]
        if not active:
            break
        cls_idx = min(active, key=lambda c: (int(class_counts[c]), int(totals[c]), c))
        pos_i = _next_unused(cls_idx)
        if pos_i is None:
            exhausted[cls_idx] = True
            continue
        _select(pos_i)

    if len(selected_pos) < target_images:
        for mask_rem in ((~used) & (~only_person), ~used):
            rem = np.where(mask_rem)[0]
            rng.shuffle(rem)
            for pos in rem:
                _select(int(pos))
                if len(selected_pos) >= target_images:
                    break
            if len(selected_pos) >= target_images:
                break

    return pool.iloc[selected_pos].sort_values("image_id").reset_index(drop=True)


def split_coco_train_tune_tables(
    df_labels: pd.DataFrame,
    classes: tuple[str, ...] = COCO_CLASSES,
    tune_max_images: int = 240,
    tune_fraction: float | None = None,
    min_per_class: int = 3,
    max_person_only: int | None = 6,
    max_class_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a balanced tune DataFrame from `df_labels` so Train and Tune have 0% overlap."""
    df_tune = build_coco_balanced_tune_table(
        df_labels,
        classes=classes,
        max_images=tune_max_images,
        tune_fraction=tune_fraction,
        min_per_class=min_per_class,
        max_person_only=max_person_only,
        max_class_fraction=max_class_fraction,
        seed=seed,
    )
    tune_ids = set(df_tune["image_id"].tolist())
    df_train = df_labels[~df_labels["image_id"].isin(tune_ids)].reset_index(drop=True)
    return df_train, df_tune

