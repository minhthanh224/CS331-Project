import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


# =====================================================================
# STEP 1: METRICS FOR MULTI-LABEL CLASSIFICATION
# =====================================================================
def compute_classification_metrics(targets, probabilities, threshold: float = 0.5):
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    else:
        targets = np.asarray(targets)

    if torch.is_tensor(probabilities):
        probabilities = probabilities.detach().cpu().numpy()
    else:
        probabilities = np.asarray(probabilities)

    targets_bin = (targets > 0.5).astype(np.int32)
    preds_bin = (probabilities >= threshold).astype(np.int32)

    ap_per_class = []
    num_classes = targets_bin.shape[1]
    for i in range(num_classes):
        if targets_bin[:, i].sum() > 0:
            ap = float(average_precision_score(targets_bin[:, i], probabilities[:, i]))
        else:
            ap = float("nan")
        ap_per_class.append(ap)

    mAP = float(np.nanmean(ap_per_class))

    macro_f1 = float(f1_score(targets_bin, preds_bin, average="macro", zero_division=0))
    micro_f1 = float(f1_score(targets_bin, preds_bin, average="micro", zero_division=0))
    macro_precision = float(precision_score(targets_bin, preds_bin, average="macro", zero_division=0))
    micro_precision = float(precision_score(targets_bin, preds_bin, average="micro", zero_division=0))
    macro_recall = float(recall_score(targets_bin, preds_bin, average="macro", zero_division=0))
    micro_recall = float(recall_score(targets_bin, preds_bin, average="micro", zero_division=0))
    exact_match = float(np.mean(np.all(targets_bin == preds_bin, axis=1)))

    return {
        "mAP": mAP,
        "ap_per_class": ap_per_class,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "exact_match": exact_match,
    }


# =====================================================================
# STEP 2: METRICS FOR SEMANTIC SEGMENTATION (WSSS: CAM / CRF / SAM)
# =====================================================================
class SegmentationEvaluator:
    def __init__(self, num_classes: int = 21, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion_matrix.fill(0)

    def update(self, preds, targets) -> None:
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        else:
            preds = np.asarray(preds)

        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        else:
            targets = np.asarray(targets)

        preds_flat = preds.astype(np.int64).ravel()
        targets_flat = targets.astype(np.int64).ravel()

        # Filter out boundary/ambiguous pixels marked with ignore_index (255 in VOC2012)
        valid_mask = (
            (targets_flat != self.ignore_index)
            & (targets_flat >= 0)
            & (targets_flat < self.num_classes)
            & (preds_flat >= 0)
            & (preds_flat < self.num_classes)
        )

        indices = self.num_classes * targets_flat[valid_mask] + preds_flat[valid_mask]
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        hist = self.confusion_matrix.astype(np.float64)
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp

        # 1. IoU & mIoU
        union = tp + fp + fn
        iou_per_class = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        miou = float(np.nanmean(iou_per_class))
        foreground_miou = float(np.nanmean(iou_per_class[1:])) if self.num_classes > 1 else miou

        # 2. Dice (Pixel F1-score)
        dice_denom = 2 * tp + fp + fn
        dice_per_class = np.divide(2 * tp, dice_denom, out=np.full_like(tp, np.nan), where=dice_denom > 0)
        mean_dice = float(np.nanmean(dice_per_class))

        # 3. Pixel Precision & Recall
        precision_per_class = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0)
        recall_per_class = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)

        # 4. Pixel Accuracy (PA) & Mean Pixel Accuracy (mPA)
        total_pixels = hist.sum()
        pixel_accuracy = float(tp.sum() / total_pixels) if total_pixels > 0 else 0.0
        mean_pixel_accuracy = float(np.nanmean(recall_per_class))

        return {
            "mIoU": miou,
            "foreground_mIoU": foreground_miou,
            "mean_dice": mean_dice,
            "pixel_accuracy": pixel_accuracy,
            "mean_pixel_accuracy": mean_pixel_accuracy,
            "mean_precision": float(np.nanmean(precision_per_class)),
            "mean_recall": float(np.nanmean(recall_per_class)),
            "iou_per_class": iou_per_class.tolist(),
            "dice_per_class": dice_per_class.tolist(),
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
        }
