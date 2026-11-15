"""Flask web demo for ResNet-101 WSSS (Multi-label Classification + CAM/LayerCAM + DenseCRF + SAM)."""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageOps
from torchvision import transforms

# Resolve project root so `src` and `.venv` site-packages are importable from any working directory.
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENV_SITE_PACKAGES = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.is_dir() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(VENV_SITE_PACKAGES))

from src.cam import compute_cams, gen_cam_pseudo_mask, select_active_classes
from src.constants import VOC_CLASSES
from src.dense_CRF import get_dense_crf_mask
from src.engine import normalize_batch
from src.models import load_classifier
from src.sam_utils import build_sam_predictor, gen_sam_pseudo_mask
from src.utils import get_device

MAX_IMAGE_SIDE = 1024
CNN_INPUT_SIZE = 448

# High-contrast 21-class palette: index 0 = background, indices 1..20 = VOC_CLASSES.
CLASS_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),        # 0: background
    (239, 68, 68),    # 1: aeroplane
    (34, 197, 94),    # 2: bicycle
    (234, 179, 8),    # 3: bird
    (59, 130, 246),   # 4: boat
    (168, 85, 247),   # 5: bottle
    (6, 182, 212),    # 6: bus
    (249, 115, 22),   # 7: car
    (236, 72, 153),   # 8: cat
    (132, 204, 22),   # 9: chair
    (20, 184, 166),   # 10: cow
    (245, 158, 11),   # 11: diningtable
    (99, 102, 241),   # 12: dog
    (244, 63, 94),    # 13: horse
    (16, 185, 129),   # 14: motorbike
    (56, 189, 248),   # 15: person
    (163, 230, 53),   # 16: pottedplant
    (251, 146, 60),   # 17: sheep
    (192, 132, 252),  # 18: sofa
    (45, 212, 191),   # 19: train
    (250, 204, 21),   # 20: tvmonitor
]

CLASS_PALETTE_NP = np.array(CLASS_PALETTE, dtype=np.uint8)

app = Flask(
    __name__,
    template_folder=str(DEMO_DIR / "templates"),
    static_folder=str(DEMO_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload cap

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
    transforms.ToTensor(),
])


class ModelBundle:
    """Thread-safe container holding ResNet-101 and optional SAM predictor."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.device = get_device()
        self.classifier = None
        self.sam_predictor = None
        self.sam_error: str | None = None
        self.resnet_ckpt: Path = Path(
            os.environ.get("RESNET101_CKPT", PROJECT_ROOT / "checkpoints" / "resnet101" / "best.pt")
        )
        self.sam_ckpt: Path = Path(
            os.environ.get("SAM_CKPT", PROJECT_ROOT / "checkpoints" / "sam_vit_b_01ec64.pth")
        )

    def ensure_loaded(self) -> None:
        with self.lock:
            if self.classifier is None:
                if not self.resnet_ckpt.is_file():
                    raise FileNotFoundError(f"Không tìm thấy checkpoint ResNet-101 tại: {self.resnet_ckpt}")
                self.classifier = load_classifier(
                    model_name="resnet101",
                    checkpoint_path=self.resnet_ckpt,
                    device=self.device,
                    num_classes=len(VOC_CLASSES),
                )

            if self.sam_predictor is None and self.sam_error is None:
                if not self.sam_ckpt.is_file():
                    self.sam_error = f"Không tìm thấy checkpoint SAM tại {self.sam_ckpt.name}"
                else:
                    try:
                        self.sam_predictor = build_sam_predictor("vit_b", self.sam_ckpt, device=self.device)
                    except Exception as exc:  # pragma: no cover - optional dependency fallback
                        self.sam_error = str(exc)


BUNDLE = ModelBundle()


def _rgb_to_data_url(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb, mode="RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _rgba_to_data_url(img_rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgba, mode="RGBA").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _cam_to_heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    cam_u8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _cam_overlay_rgb(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heat = _cam_to_heatmap_rgb(cam)
    blended = (1.0 - alpha) * img_rgb.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _mask_to_rgba(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) class index mask into RGBA PNG where background (0) is transparent."""
    safe_mask = np.clip(mask, 0, len(CLASS_PALETTE_NP) - 1)
    rgb = CLASS_PALETTE_NP[safe_mask]
    alpha = np.where(safe_mask > 0, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def _mask_stats(mask: np.ndarray) -> list[dict]:
    total_pixels = max(1, int(mask.size))
    unique_ids, counts = np.unique(mask, return_counts=True)
    items = []
    for cls_id, cnt in zip(unique_ids.tolist(), counts.tolist()):
        if cls_id == 0 or cls_id > len(VOC_CLASSES):
            continue
        r, g, b = CLASS_PALETTE[cls_id]
        items.append({
            "class_id": cls_id,
            "name": VOC_CLASSES[cls_id - 1],
            "color": f"#{r:02x}{g:02x}{b:02x}",
            "pixel_percent": round(100.0 * cnt / total_pixels, 2),
        })
    items.sort(key=lambda x: x["pixel_percent"], reverse=True)
    return items


def _prepare_pil_image(raw_bytes: bytes) -> Image.Image:
    pil_img = Image.open(io.BytesIO(raw_bytes))
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    w, h = pil_img.size
    max_side = max(w, h)
    if max_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(max_side)
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        pil_img = pil_img.resize(new_size, resample=Image.Resampling.BILINEAR)
    return pil_img


def _parse_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(val: str | None, default: float, lo: float, hi: float) -> float:
    try:
        parsed = float(val) if val is not None else default
    except ValueError:
        return default
    return max(lo, min(hi, parsed))


def _discover_sample_images(limit: int = 6) -> list[dict]:
    voc_img_dir = PROJECT_ROOT / "data" / "VOC2012" / "JPEGImages"
    val_txt = PROJECT_ROOT / "data" / "VOC2012" / "ImageSets" / "Segmentation" / "val.txt"
    if not voc_img_dir.is_dir():
        return []

    sample_ids: list[str] = []
    if val_txt.is_file():
        lines = [line.strip() for line in val_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Pick evenly spaced IDs across val set for diverse scenes.
        if lines:
            step = max(1, len(lines) // limit)
            for idx in range(0, len(lines), step):
                candidate = lines[idx]
                if (voc_img_dir / f"{candidate}.jpg").is_file():
                    sample_ids.append(candidate)
                if len(sample_ids) >= limit:
                    break

    if not sample_ids:
        for p in sorted(voc_img_dir.glob("*.jpg"))[:limit]:
            sample_ids.append(p.stem)

    return [{"id": img_id, "filename": f"{img_id}.jpg", "url": f"/api/samples/{img_id}"} for img_id in sample_ids]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    BUNDLE.ensure_loaded()
    return jsonify({
        "backbone": "resnet101",
        "device": str(BUNDLE.device),
        "checkpoint": BUNDLE.resnet_ckpt.name,
        "sam_available": BUNDLE.sam_predictor is not None,
        "sam_error": BUNDLE.sam_error,
        "classes": list(VOC_CLASSES),
        "max_image_side": MAX_IMAGE_SIDE,
    })


@app.route("/api/samples", methods=["GET"])
def api_samples():
    return jsonify({"samples": _discover_sample_images()})


@app.route("/api/samples/<image_id>", methods=["GET"])
def api_sample_file(image_id: str):
    if not image_id.replace("_", "").isalnum():
        return jsonify({"error": "Invalid sample ID"}), 400
    img_path = PROJECT_ROOT / "data" / "VOC2012" / "JPEGImages" / f"{image_id}.jpg"
    if not img_path.is_file():
        return jsonify({"error": "Sample not found"}), 404
    return send_file(img_path, mimetype="image/jpeg")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "Vui lòng chọn hoặc kéo thả một ảnh."}), 400

    file = request.files["image"]
    raw_bytes = file.read()
    if not raw_bytes:
        return jsonify({"error": "Tệp ảnh rỗng."}), 400

    try:
        pil_img = _prepare_pil_image(raw_bytes)
    except Exception as exc:
        return jsonify({"error": f"Không thể đọc ảnh: {exc}"}), 400

    cam_method = request.form.get("cam_method", "cam").strip().lower()
    if cam_method not in {"cam", "layercam"}:
        cam_method = "cam"
    use_tta = _parse_bool(request.form.get("use_tta"), default=False)
    use_crf = _parse_bool(request.form.get("use_crf"), default=True)
    use_sam = _parse_bool(request.form.get("use_sam"), default=True)
    cls_threshold = _parse_float(request.form.get("cls_threshold"), 0.5, 0.05, 0.95)
    cam_threshold = _parse_float(request.form.get("cam_threshold"), 0.25, 0.05, 0.9)
    crf_alpha = _parse_float(request.form.get("crf_alpha"), 2.0, 0.5, 10.0)

    BUNDLE.ensure_loaded()
    img_u8 = np.array(pil_img, dtype=np.uint8)
    orig_h, orig_w = img_u8.shape[:2]

    input_tensor = _EVAL_TRANSFORM(pil_img).unsqueeze(0).to(BUNDLE.device)
    scales = (0.75, 1.0, 1.25) if use_tta else (1.0,)
    layers = ("layer2", "layer3", "layer4") if cam_method == "layercam" else None

    timings: dict[str, float] = {}
    t_total_start = time.perf_counter()

    with BUNDLE.lock:
        # 1. Multi-label classification
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = BUNDLE.classifier(normalize_batch(input_tensor))
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        timings["cls_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

        passed_threshold = np.where(probs >= cls_threshold)[0]
        fallback_used = len(passed_threshold) == 0
        active_indices = select_active_classes(probs, threshold=cls_threshold)

        # 2. CAM / LayerCAM generation at native/display resolution
        t0 = time.perf_counter()
        cams = compute_cams(
            BUNDLE.classifier,
            input_tensor,
            active_indices,
            method=cam_method,
            target_size=(orig_h, orig_w),
            scales=scales,
            hflip=use_tta,
            layers=layers,
        )
        timings["cam_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

        # 3. Vanilla CAM pseudo-mask
        vanilla_mask = gen_cam_pseudo_mask(cams, active_indices, cam_threshold=cam_threshold)

        # 4. CAM + DenseCRF mask
        crf_mask = None
        if use_crf:
            t0 = time.perf_counter()
            crf_mask = get_dense_crf_mask(
                image_rgb=img_u8,
                cams=cams,
                class_indices=active_indices,
                alpha=crf_alpha,
            )
            timings["crf_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

        # 5. CAM + SAM mask
        sam_mask = None
        if use_sam and BUNDLE.sam_predictor is not None:
            t0 = time.perf_counter()
            sam_mask = gen_sam_pseudo_mask(
                predictor=BUNDLE.sam_predictor,
                image_rgb=img_u8,
                cams=cams,
                class_indices=active_indices,
                cam_threshold=cam_threshold,
                bg_threshold=0.05,
                use_neg_points=True,
                box_pad=24,
                encode_image=True,
            )
            timings["sam_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

    timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000.0, 1)

    # Build classification summary sorted by probability descending
    probs_list = []
    active_set = set(active_indices.tolist())
    for idx, cls_name in enumerate(VOC_CLASSES):
        r, g, b = CLASS_PALETTE[idx + 1]
        probs_list.append({
            "class_idx": idx,
            "class_id": idx + 1,
            "name": cls_name,
            "prob": round(float(probs[idx]), 4),
            "active": idx in active_set,
            "above_threshold": bool(probs[idx] >= cls_threshold),
            "color": f"#{r:02x}{g:02x}{b:02x}",
        })
    probs_list.sort(key=lambda item: item["prob"], reverse=True)

    combined_cam = cams.max(axis=0)
    per_class_cams = {
        VOC_CLASSES[int(cls_idx)]: _rgb_to_data_url(_cam_overlay_rgb(img_u8, cams[k], alpha=0.52))
        for k, cls_idx in enumerate(active_indices)
    }

    primary_mask = sam_mask if sam_mask is not None else (crf_mask if crf_mask is not None else vanilla_mask)

    response = {
        "image_size": {"width": orig_w, "height": orig_h},
        "device": str(BUNDLE.device),
        "cam_method": cam_method,
        "fallback_used": fallback_used,
        "probs": probs_list,
        "active_classes": [VOC_CLASSES[int(i)] for i in active_indices],
        "timings": timings,
        "images": {
            "original": _rgb_to_data_url(img_u8),
            "cam_overlay": _rgb_to_data_url(_cam_overlay_rgb(img_u8, combined_cam, alpha=0.52)),
            "cam_heatmap": _rgb_to_data_url(_cam_to_heatmap_rgb(combined_cam)),
            "per_class_cams": per_class_cams,
            "vanilla_mask": _rgba_to_data_url(_mask_to_rgba(vanilla_mask)),
            "crf_mask": _rgba_to_data_url(_mask_to_rgba(crf_mask)) if crf_mask is not None else None,
            "sam_mask": _rgba_to_data_url(_mask_to_rgba(sam_mask)) if sam_mask is not None else None,
        },
        "legend": _mask_stats(primary_mask),
        "vanilla_legend": _mask_stats(vanilla_mask),
        "crf_legend": _mask_stats(crf_mask) if crf_mask is not None else [],
        "sam_legend": _mask_stats(sam_mask) if sam_mask is not None else [],
    }
    return jsonify(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet-101 WSSS Web Demo")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override path to ResNet-101 checkpoint")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint is not None:
        BUNDLE.resnet_ckpt = args.checkpoint.resolve()
    BUNDLE.ensure_loaded()
    print(
        f"ResNet-101 Web Demo ready on http://{args.host}:{args.port} | "
        f"device={BUNDLE.device} | sam_available={BUNDLE.sam_predictor is not None}"
    )
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)

