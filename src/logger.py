from dataclasses import asdict, is_dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


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

