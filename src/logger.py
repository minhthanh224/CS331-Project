from dataclasses import asdict, is_dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from .constants import VOC_CLASSES  # re-exported: `from src.logger import VOC_CLASSES` keeps working


def _serialize_config(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        raw = asdict(config)
    elif isinstance(config, dict):
        raw = dict(config)
    elif hasattr(config, "__dict__"):
        raw = dict(vars(config))
    else:
        return {"value": str(config)}

    serialized = {}
    for k, v in raw.items():
        if isinstance(v, Path):
            serialized[k] = str(v)
        elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            serialized[k] = v
        else:
            serialized[k] = str(v)
    return serialized


def log_eval_results(
    experiment_name: str,
    metrics: Dict[str, Any],
    config: Optional[Union[Dict[str, Any], Any]] = None,
    task: str = "segmentation",
    log_dir: Union[str, Path] = "runs",
    classes: Sequence[str] = VOC_CLASSES,
    print_summary: bool = True,
) -> Dict[str, Path]:
    """
    Log evaluation results (Classification or Segmentation) with timestamp and configuration.

    Automatically updates 3 files in `log_dir`:
    1. `eval_history.jsonl`: Stores full metrics (including per-class IoU/AP), configuration, and timestamp.
    2. `segmentation_summary.csv` (or `classification_summary.csv`): Summary table for Excel/Pandas.
    3. `eval_report.md`: Formatted Markdown report ready to copy into project documentation.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    config_dict = _serialize_config(config)

    # Automatically infer task type if not explicitly specified
    if "mIoU" in metrics:
        task = "segmentation"
    elif "mAP" in metrics:
        task = "classification"

    record = {
        "timestamp": timestamp_str,
        "experiment_name": experiment_name,
        "task": task,
        "config": config_dict,
        "metrics": metrics,
    }

    # 1. Append full record to JSONL history file
    jsonl_path = log_dir / "eval_history.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. Append summary row to task-specific CSV file
    csv_path = log_dir / f"{task}_summary.csv"
    if task == "segmentation":
        row = {
            "timestamp": timestamp_str,
            "experiment_name": experiment_name,
            "mIoU": round(float(metrics.get("mIoU", 0.0)), 4),
            "foreground_mIoU": round(float(metrics.get("foreground_mIoU", 0.0)), 4),
            "mean_dice": round(float(metrics.get("mean_dice", 0.0)), 4),
            "pixel_accuracy": round(float(metrics.get("pixel_accuracy", 0.0)), 4),
            "mean_pixel_accuracy": round(float(metrics.get("mean_pixel_accuracy", 0.0)), 4),
            "mean_precision": round(float(metrics.get("mean_precision", 0.0)), 4),
            "mean_recall": round(float(metrics.get("mean_recall", 0.0)), 4),
            "config": json.dumps(config_dict, ensure_ascii=False),
        }
    else:
        row = {
            "timestamp": timestamp_str,
            "experiment_name": experiment_name,
            "mAP": round(float(metrics.get("mAP", 0.0)), 4),
            "macro_f1": round(float(metrics.get("macro_f1", 0.0)), 4),
            "micro_f1": round(float(metrics.get("micro_f1", 0.0)), 4),
            "macro_precision": round(float(metrics.get("macro_precision", 0.0)), 4),
            "micro_precision": round(float(metrics.get("micro_precision", 0.0)), 4),
            "macro_recall": round(float(metrics.get("macro_recall", 0.0)), 4),
            "micro_recall": round(float(metrics.get("micro_recall", 0.0)), 4),
            "exact_match": round(float(metrics.get("exact_match", 0.0)), 4),
            "config": json.dumps(config_dict, ensure_ascii=False),
        }

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # 3. Append human-readable entry to Markdown evaluation report
    md_path = log_dir / "eval_report.md"
    with md_path.open("a", encoding="utf-8") as f:
        f.write(f"## [{timestamp_str}] {experiment_name} ({task.upper()})\n\n")
        if config_dict:
            config_items = ", ".join(f"`{k}={v}`" for k, v in config_dict.items())
            f.write(f"- **Cấu hình**: {config_items}\n")

        if task == "segmentation":
            f.write(
                f"- **Kết quả chính**: "
                f"`mIoU = {metrics.get('mIoU', 0.0):.4f}` | "
                f"`Foreground mIoU = {metrics.get('foreground_mIoU', 0.0):.4f}` | "
                f"`Mean Dice = {metrics.get('mean_dice', 0.0):.4f}` | "
                f"`PA = {metrics.get('pixel_accuracy', 0.0):.2%}` | "
                f"`Precision = {metrics.get('mean_precision', 0.0):.4f}` | "
                f"`Recall = {metrics.get('mean_recall', 0.0):.4f}`\n\n"
            )
            if "iou_per_class" in metrics:
                voc_names = ("background", *classes)
                f.write("<details><summary>IoU chi tiết từng lớp</summary>\n\n")
                f.write("| Lớp | IoU |\n| :--- | :---: |\n")
                for cls_name, iou_val in zip(voc_names, metrics["iou_per_class"]):
                    f.write(f"| {cls_name} | {iou_val:.4f} |\n")
                f.write("\n</details>\n\n")
        else:
            f.write(
                f"- **Kết quả chính**: "
                f"`mAP = {metrics.get('mAP', 0.0):.4f}` | "
                f"`Macro F1 = {metrics.get('macro_f1', 0.0):.4f}` | "
                f"`Micro F1 = {metrics.get('micro_f1', 0.0):.4f}` | "
                f"`Exact Match = {metrics.get('exact_match', 0.0):.2%}`\n\n"
            )
            if "ap_per_class" in metrics:
                f.write("<details><summary>AP chi tiết từng lớp</summary>\n\n")
                f.write("| Lớp | AP |\n| :--- | :---: |\n")
                for cls_name, ap_val in zip(classes, metrics["ap_per_class"]):
                    f.write(f"| {cls_name} | {ap_val:.4f} |\n")
                f.write("\n</details>\n\n")
        f.write("---\n\n")

    if print_summary:
        print(f"[Logger] Đã lưu kết quả '{experiment_name}' ({timestamp_str}) vào: {log_dir.resolve()}")

    return {
        "jsonl": jsonl_path,
        "csv": csv_path,
        "markdown": md_path,
    }



# =====================================================================
# Console printing and summary-table rows (previously copy-pasted in every notebook)
# =====================================================================
_SEG_ROW_FIELDS = (
    # (metric key, short label for percent=False, label for percent=True)
    ("mIoU", "mIoU", "mIoU (%)"),
    ("foreground_mIoU", "Fg mIoU", "Fg mIoU (%)"),
    ("mean_dice", "Mean Dice", "Mean Dice (%)"),
    ("pixel_accuracy", "Pixel Acc", "Pixel Acc (%)"),
    ("mean_precision", "Precision", "Precision (%)"),
    ("mean_recall", "Recall", "Recall (%)"),
)

_CLS_ROW_FIELDS = (
    ("mAP", "mAP", "mAP (%)"),
    ("macro_f1", "Macro F1", "Macro F1 (%)"),
    ("micro_f1", "Micro F1", "Micro F1 (%)"),
    ("macro_precision", "Macro Precision", "Macro Precision (%)"),
    ("macro_recall", "Macro Recall", "Macro Recall (%)"),
    ("exact_match", "Exact Match", "Exact Match (%)"),
)


def _metrics_row(metrics: Dict[str, Any], fields, percent: bool, extra: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(extra)
    for key, short_label, percent_label in fields:
        value = float(metrics[key])
        if percent:
            row[percent_label] = round(value * 100, 2)
        else:
            row[short_label] = round(value, 4)
    return row


def seg_summary_row(metrics: Dict[str, Any], percent: bool = True, **extra: Any) -> Dict[str, Any]:
    """One comparison-table row for segmentation metrics; `extra` (e.g. Backbone=..., Method=...) come first.

    percent=True -> columns like "mIoU (%)" rounded to 2 decimals; False -> "mIoU" rounded to 4 decimals.
    """
    return _metrics_row(metrics, _SEG_ROW_FIELDS, percent, extra)


def cls_summary_row(metrics: Dict[str, Any], percent: bool = True, **extra: Any) -> Dict[str, Any]:
    """One comparison-table row for multi-label classification metrics (see seg_summary_row)."""
    return _metrics_row(metrics, _CLS_ROW_FIELDS, percent, extra)


def print_seg_metrics(
    metrics: Dict[str, Any],
    title: str = "SEGMENTATION RESULTS",
    classes: Sequence[str] = VOC_CLASSES,
    per_class: bool = False,
) -> None:
    print(f"=== {title} ===")
    print(f"mIoU (all classes incl. background) : {metrics['mIoU']:.4f}")
    print(f"Foreground mIoU (object classes)    : {metrics['foreground_mIoU']:.4f}")
    print(f"Mean Dice (Pixel F1)                : {metrics['mean_dice']:.4f}")
    print(f"Pixel Accuracy (PA)                 : {metrics['pixel_accuracy']:.2%}")
    print(f"Mean Pixel Accuracy (mPA)           : {metrics['mean_pixel_accuracy']:.4f}")
    print(f"Mean Pixel Precision                : {metrics['mean_precision']:.4f}")
    print(f"Mean Pixel Recall                   : {metrics['mean_recall']:.4f}")
    if per_class and "iou_per_class" in metrics:
        print("\nPer-class IoU:")
        for name, value in zip(("background", *classes), metrics["iou_per_class"]):
            print(f"- {name:15s}: {value:.4f}")


def print_cls_metrics(
    metrics: Dict[str, Any],
    title: str = "MULTI-LABEL CLASSIFICATION RESULTS",
    classes: Sequence[str] = VOC_CLASSES,
    per_class: bool = False,
) -> None:
    print(f"=== {title} ===")
    print(f"mAP             : {metrics['mAP']:.4f}")
    print(f"Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"Micro F1        : {metrics['micro_f1']:.4f}")
    print(f"Macro Precision : {metrics['macro_precision']:.4f}")
    print(f"Micro Precision : {metrics['micro_precision']:.4f}")
    print(f"Macro Recall    : {metrics['macro_recall']:.4f}")
    print(f"Micro Recall    : {metrics['micro_recall']:.4f}")
    print(f"Exact Match     : {metrics['exact_match']:.2%}")
    if per_class and "ap_per_class" in metrics:
        print("\nPer-class AP:")
        for name, value in zip(classes, metrics["ap_per_class"]):
            print(f"- {name:15s}: {value:.4f}")
