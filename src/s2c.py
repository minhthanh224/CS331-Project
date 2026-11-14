"""S2C (From SAM to CAMs) training utilities for Weakly Supervised Semantic Segmentation.

Implements the two SAM-to-CAM distillation mechanisms on top of standard multi-label
classification (`BCEWithLogitsLoss`):
1. SSC (SAM-Segment Contrasting): feature-level prototype InfoNCE loss using offline-cached
   SAM Automatic Mask Generator (AMG) segment maps.
2. CPM (CAM-based Prompting Module): logit-level pixel cross-entropy loss using SAM pseudo-masks
   prompted from the classifier's CAM or LayerCAM heatmaps.
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops import batched_nms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from .cam import compute_cams
from .constants import IGNORE_INDEX, VOC_CLASSES
from .engine import normalize_batch
from .models import default_layercam_layers, get_classifier_head
from .sam_utils import gen_sam_pseudo_mask


class S2CProjectionHead(nn.Module):
    """Lightweight 1x1 conv projection head for SSC contrastive training (discarded at inference)."""

    def __init__(self, in_channels: int = 2048, proj_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim, kernel_size=1, bias=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


def forward_resnet_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a torchvision ResNet forward pass returning `(logits, layer4_features)`."""
    if not hasattr(model, "layer4") or not hasattr(model, "fc"):
        raise ValueError(f"forward_resnet_features requires a ResNet model, got {type(model).__name__}")
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    feat = model.layer4(x)
    pooled = model.avgpool(feat)
    logits = model.fc(torch.flatten(pooled, 1))
    return logits, feat


def rasterize_sam_masks(
    anns: list[dict],
    height: int,
    width: int,
    max_segments: int = 255,
) -> np.ndarray:
    """Convert SAM AMG mask dicts into a `(height, width)` uint8 segment ID map.

    ID `0` marks pixels not covered by any SAM segment. Masks are painted from largest
    to smallest area so fine object parts keep distinct segment IDs instead of being
    overwritten by large background masks.
    """
    seg_map = np.zeros((height, width), dtype=np.uint8)
    if not anns:
        return seg_map

    sorted_anns = sorted(anns, key=lambda item: int(item.get("area", 0)), reverse=True)
    if len(sorted_anns) > max_segments:
        # Keep the largest/highest-confidence masks up to max_segments while preserving draw order
        top_anns = sorted(
            sorted_anns,
            key=lambda item: float(item.get("predicted_iou", 0.0)) * np.log1p(float(item.get("area", 1))),
            reverse=True,
        )[:max_segments]
        sorted_anns = sorted(top_anns, key=lambda item: int(item.get("area", 0)), reverse=True)

    for seg_id, ann in enumerate(sorted_anns, start=1):
        mask = np.asarray(ann["segmentation"], dtype=bool)
        if mask.shape != (height, width):
            continue
        seg_map[mask] = seg_id
    return seg_map


def build_sam_amg(
    sam_model,
    points_per_side: int = 16,
    points_per_batch: int = 128,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 100,
    crop_n_layers: int = 0,
):
    """Build a `SamAutomaticMaskGenerator` configured for fast WSSS segment caching."""
    from segment_anything import SamAutomaticMaskGenerator

    return SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        min_mask_region_area=min_mask_region_area,
    )


def _auto_sam_amp_dtype(device: torch.device, use_amp: bool = True) -> torch.dtype | None:
    """Pick BF16 on Ada/Ampere+ GPUs (e.g. L4), FP16 on older CUDA GPUs, or None on CPU."""
    if not use_amp or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _save_uint8_png(arr: np.ndarray, out_path: Path) -> None:
    Image.fromarray(arr, mode="L").save(out_path, compress_level=1)


def _encode_sam_images_batch(
    sam_model,
    images_u8: list[np.ndarray],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]]]:
    """Batch-run SAM's heavy ViT image_encoder in AMP, returning FP32 embeddings `(B, 256, 64, 64)`."""
    from segment_anything.utils.transforms import ResizeLongestSide

    resize_tf = ResizeLongestSide(sam_model.image_encoder.img_size)
    prepped = []
    orig_sizes: list[tuple[int, int]] = []
    input_sizes: list[tuple[int, int]] = []

    for img_u8 in images_u8:
        H, W = img_u8.shape[:2]
        orig_sizes.append((H, W))
        resized = resize_tf.apply_image(img_u8)
        input_sizes.append(resized.shape[:2])
        t = torch.as_tensor(resized, dtype=torch.float32, device=device).permute(2, 0, 1)
        prepped.append(sam_model.preprocess(t))

    batch_t = torch.stack(prepped, dim=0)
    with torch.inference_mode():
        if amp_dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                features = sam_model.image_encoder(batch_t)
        else:
            features = sam_model.image_encoder(batch_t)
    return features.float(), orig_sizes, input_sizes


def _decode_amg_segment_map_gpu(
    sam_model,
    image_embedding: torch.Tensor,
    original_size: tuple[int, int],
    input_size: tuple[int, int],
    points_norm: np.ndarray,
    points_per_batch: int = 256,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    box_nms_thresh: float = 0.7,
    max_segments: int = 255,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Decode SAM grid prompts directly to a `(H, W)` uint8 segment map on GPU without RLE round-trips."""
    from segment_anything.utils.amg import batched_mask_to_box, calculate_stability_score

    device = image_embedding.device
    H, W = original_size
    H_in, W_in = input_size
    max_hw = max(H, W)
    H_256 = max(1, int(round(256.0 * H / max_hw)))
    W_256 = max(1, int(round(256.0 * W / max_hw)))

    pts_in = points_norm * np.array([W_in, H_in], dtype=np.float32)
    pts_t = torch.as_tensor(pts_in, dtype=torch.float32, device=device).unsqueeze(1)
    lbls_t = torch.ones((len(pts_t), 1), dtype=torch.int32, device=device)
    dense_pe = sam_model.prompt_encoder.get_dense_pe()

    mask_chunks = []
    iou_chunks = []
    with torch.inference_mode():
        for start in range(0, len(pts_t), points_per_batch):
            pts_c = pts_t[start : start + points_per_batch]
            lbls_c = lbls_t[start : start + points_per_batch]
            if amp_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    sparse_emb, dense_emb = sam_model.prompt_encoder(
                        points=(pts_c, lbls_c), boxes=None, masks=None
                    )
                    low_masks, iou_preds = sam_model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )
            else:
                sparse_emb, dense_emb = sam_model.prompt_encoder(
                    points=(pts_c, lbls_c), boxes=None, masks=None
                )
                low_masks, iou_preds = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                )

            flat_masks = low_masks.float().flatten(0, 1)
            flat_ious = iou_preds.float().flatten(0, 1)
            keep = flat_ious > pred_iou_thresh
            if keep.any():
                mask_chunks.append(flat_masks[keep])
                iou_chunks.append(flat_ious[keep])

    if not mask_chunks:
        return np.zeros((H, W), dtype=np.uint8)

    masks = torch.cat(mask_chunks, dim=0)
    scores = torch.cat(iou_chunks, dim=0)

    stability = calculate_stability_score(
        masks, mask_threshold=sam_model.mask_threshold, threshold_offset=1.0
    )
    keep_stab = stability >= stability_score_thresh
    if not keep_stab.any():
        return np.zeros((H, W), dtype=np.uint8)

    masks = masks[keep_stab]
    scores = scores[keep_stab]

    # Unpad & resize surviving masks to (H_256, W_256) on GPU
    masks = sam_model.postprocess_masks(
        masks.unsqueeze(1), input_size=input_size, original_size=(H_256, W_256)
    ).squeeze(1)
    bin_masks = masks > sam_model.mask_threshold
    areas = bin_masks.flatten(1).sum(dim=1)
    non_empty = areas > 0
    if not non_empty.any():
        return np.zeros((H, W), dtype=np.uint8)

    bin_masks = bin_masks[non_empty]
    scores = scores[non_empty]
    boxes = batched_mask_to_box(bin_masks)

    keep_nms = batched_nms(
        boxes.float(),
        scores.float(),
        torch.zeros(len(boxes), dtype=torch.long, device=device),
        iou_threshold=box_nms_thresh,
    )
    bin_masks = bin_masks[keep_nms]
    scores = scores[keep_nms]
    areas = bin_masks.flatten(1).sum(dim=1)

    if len(bin_masks) > max_segments:
        rank = scores * torch.log1p(areas.float())
        top_idx = torch.topk(rank, k=max_segments).indices
        bin_masks = bin_masks[top_idx]
        areas = areas[top_idx]

    # Paint largest-to-smallest so fine object parts keep distinct segment IDs
    order = torch.argsort(areas, descending=True)
    seg_256 = torch.zeros((H_256, W_256), dtype=torch.uint8, device=device)
    for seg_id, idx_m in enumerate(order.tolist(), start=1):
        seg_256[bin_masks[idx_m]] = seg_id

    if (H_256, W_256) == (H, W):
        return seg_256.cpu().numpy()
    seg_full = (
        F.interpolate(seg_256[None, None].float(), size=(H, W), mode="nearest")
        .squeeze(0)
        .squeeze(0)
        .to(torch.uint8)
    )
    return seg_full.cpu().numpy()


def cache_sam_segments(
    voc_root: Path,
    image_ids: Sequence[str],
    amg=None,
    output_dir: Path = Path("sam_segments"),
    sam_model=None,
    encode_batch_size: int = 12,
    points_per_side: int = 16,
    points_per_batch: int = 256,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    box_nms_thresh: float = 0.7,
    max_segments: int = 255,
    use_amp: bool = True,
    num_io_workers: int = 4,
    desc: str = "Caching SAM AMG segments",
) -> dict[str, int]:
    """Pre-compute and save SAM segment maps as uint8 PNGs (`{img_id}.png`), skipping existing ones.

    When `sam_model` is provided (or inferred from `amg.predictor.model`), uses batched AMP
    image encoding + GPU-only mask decoding/NMS/rasterization + async PNG writes for 4-8x speedup.
    """
    voc_root = Path(voc_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = voc_root / "JPEGImages"

    pending_ids: list[str] = []
    skipped = 0
    for img_id in image_ids:
        if (output_dir / f"{img_id}.png").is_file():
            skipped += 1
        else:
            pending_ids.append(img_id)

    if not pending_ids:
        return {"generated": 0, "skipped": skipped, "total": len(image_ids)}

    if sam_model is None and amg is not None and hasattr(amg, "predictor") and hasattr(amg.predictor, "model"):
        sam_model = amg.predictor.model

    if sam_model is None:
        if amg is None:
            raise ValueError("Provide either `sam_model` or `amg` to cache_sam_segments")
        generated = 0
        for img_id in tqdm(pending_ids, desc=desc):
            with Image.open(image_dir / f"{img_id}.jpg") as source:
                img_u8 = np.array(source.convert("RGB"), dtype=np.uint8)
            H, W = img_u8.shape[:2]
            with torch.inference_mode():
                anns = amg.generate(img_u8)
            seg_map = rasterize_sam_masks(anns, H, W, max_segments=max_segments)
            _save_uint8_png(seg_map, output_dir / f"{img_id}.png")
            generated += 1
        return {"generated": generated, "skipped": skipped, "total": len(image_ids)}

    from segment_anything.utils.amg import build_all_layer_point_grids

    device = next(sam_model.parameters()).device
    amp_dtype = _auto_sam_amp_dtype(device, use_amp=use_amp)
    points_norm = build_all_layer_point_grids(points_per_side, 0, 1)[0]

    def _load_rgb(img_id: str) -> tuple[str, np.ndarray]:
        with Image.open(image_dir / f"{img_id}.jpg") as source:
            return img_id, np.array(source.convert("RGB"), dtype=np.uint8)

    generated = 0
    pbar = tqdm(total=len(pending_ids), desc=desc)
    with ThreadPoolExecutor(max_workers=max(1, num_io_workers)) as io_pool:
        save_futures = []
        for start in range(0, len(pending_ids), max(1, encode_batch_size)):
            batch_ids = pending_ids[start : start + encode_batch_size]
            loaded = list(io_pool.map(_load_rgb, batch_ids))
            images_u8 = [item[1] for item in loaded]

            features, orig_sizes, input_sizes = _encode_sam_images_batch(
                sam_model, images_u8, device=device, amp_dtype=amp_dtype
            )
            for b, (img_id, _) in enumerate(loaded):
                seg_map = _decode_amg_segment_map_gpu(
                    sam_model=sam_model,
                    image_embedding=features[b : b + 1],
                    original_size=orig_sizes[b],
                    input_size=input_sizes[b],
                    points_norm=points_norm,
                    points_per_batch=points_per_batch,
                    pred_iou_thresh=pred_iou_thresh,
                    stability_score_thresh=stability_score_thresh,
                    box_nms_thresh=box_nms_thresh,
                    max_segments=max_segments,
                    amp_dtype=amp_dtype,
                )
                save_futures.append(io_pool.submit(_save_uint8_png, seg_map, output_dir / f"{img_id}.png"))
                generated += 1
                pbar.update(1)

            # Drain completed save tasks per batch to keep pending CPU memory bounded
            if len(save_futures) >= max(4, encode_batch_size * 2):
                for fut in save_futures:
                    fut.result()
                save_futures.clear()

        for fut in save_futures:
            fut.result()
    pbar.close()

    return {"generated": generated, "skipped": skipped, "total": len(image_ids)}


def generate_cpm_pseudo_masks(
    model: nn.Module,
    predictor,
    voc_root: Path,
    labels_df: pd.DataFrame,
    output_dir: Path,
    classes: tuple[str, ...] = VOC_CLASSES,
    image_size: int = 448,
    cam_method: str = "layercam",
    layercam_layers: tuple[str, ...] | None = None,
    shallow_gamma: float = 2.0,
    cam_scales: tuple[float, ...] = (1.0,),
    cam_hflip: bool = False,
    sam_params: dict | None = None,
    ambiguous_cam_thr: float = 0.50,
    overwrite: bool = True,
    encode_batch_size: int = 12,
    use_amp: bool = True,
    num_io_workers: int = 4,
    desc: str = "Generating CPM pseudo-masks",
) -> dict[str, int]:
    """Generate SAM-refined pseudo-masks prompted by CAM or LayerCAM using image-level GT labels.

    Batches SAM's `image_encoder` in AMP when `predictor.model` is available, while keeping CAM
    extraction and mask selection exact. Pixels where SAM predicts background (`0`) despite high
    CAM confidence (`>= ambiguous_cam_thr`) are marked `IGNORE_INDEX (255)`.
    """
    voc_root = Path(voc_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = voc_root / "JPEGImages"

    if layercam_layers is None and cam_method == "layercam":
        layercam_layers = default_layercam_layers(model)

    eval_resize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    sam_cfg = dict(sam_params or {})
    sam_cfg.setdefault("cam_threshold", 0.22)
    sam_cfg.setdefault("low_threshold", 0.10)
    sam_cfg.setdefault("bg_threshold", 0.04)
    sam_cfg.setdefault("use_neg_points", True)
    sam_cfg.setdefault("box_pad", 20)

    was_training = model.training
    model.eval()

    image_ids = labels_df["image_id"].tolist()
    label_matrix = labels_df[list(classes)].to_numpy(dtype=np.float32)

    pending: list[tuple[int, str]] = []
    skipped = 0
    for idx, img_id in enumerate(image_ids):
        if not overwrite and (output_dir / f"{img_id}.png").is_file():
            skipped += 1
        else:
            pending.append((idx, img_id))

    if not pending:
        if was_training:
            model.train()
        return {"generated": 0, "skipped": skipped, "total": len(image_ids)}

    sam_model = getattr(predictor, "model", None)
    can_batch_sam = sam_model is not None and hasattr(sam_model, "image_encoder")
    if can_batch_sam:
        sam_device = next(sam_model.parameters()).device
        amp_dtype = _auto_sam_amp_dtype(sam_device, use_amp=use_amp)

    def _load_item(item: tuple[int, str]):
        idx, img_id = item
        with Image.open(image_dir / f"{img_id}.jpg") as source:
            rgb_pil = source.convert("RGB")
            img_u8 = np.array(rgb_pil, dtype=np.uint8)
            img_tensor = eval_resize(rgb_pil)
        return idx, img_id, img_u8, img_tensor

    generated = 0
    pbar = tqdm(total=len(pending), desc=desc)
    with ThreadPoolExecutor(max_workers=max(1, num_io_workers)) as io_pool:
        save_futures = []
        batch_sz = max(1, encode_batch_size) if can_batch_sam else 1
        for start in range(0, len(pending), batch_sz):
            chunk = pending[start : start + batch_sz]
            loaded = list(io_pool.map(_load_item, chunk))

            if can_batch_sam:
                # Only encode images that actually have >= 1 active GT class
                encode_positions = [
                    pos for pos, (idx, _, _, _) in enumerate(loaded)
                    if (label_matrix[idx] > 0.5).any()
                ]
                encoded_map = {}
                if encode_positions:
                    imgs_to_encode = [loaded[pos][2] for pos in encode_positions]
                    feats, orig_sizes, input_sizes = _encode_sam_images_batch(
                        sam_model, imgs_to_encode, device=sam_device, amp_dtype=amp_dtype
                    )
                    for k_pos, orig_pos in enumerate(encode_positions):
                        encoded_map[orig_pos] = (
                            feats[k_pos : k_pos + 1],
                            orig_sizes[k_pos],
                            input_sizes[k_pos],
                        )

            for pos, (idx, img_id, img_u8, img_tensor) in enumerate(loaded):
                H, W = img_u8.shape[:2]
                gt_classes = np.where(label_matrix[idx] > 0.5)[0].astype(np.int64)
                if len(gt_classes) == 0:
                    save_futures.append(
                        io_pool.submit(_save_uint8_png, np.zeros((H, W), dtype=np.uint8), output_dir / f"{img_id}.png")
                    )
                    generated += 1
                    pbar.update(1)
                    continue

                cams = compute_cams(
                    model,
                    img_tensor,
                    gt_classes,
                    method=cam_method,
                    target_size=(H, W),
                    scales=cam_scales,
                    hflip=cam_hflip,
                    layers=layercam_layers,
                    shallow_gamma=shallow_gamma,
                )
                img_rgb = img_u8.astype(np.float32) / 255.0

                if can_batch_sam:
                    feat_b, orig_sz, input_sz = encoded_map[pos]
                    predictor.features = feat_b
                    predictor.original_size = orig_sz
                    predictor.input_size = input_sz
                    predictor.is_image_set = True
                    encode_flag = False
                else:
                    encode_flag = True

                with torch.inference_mode():
                    pseudo_mask = gen_sam_pseudo_mask(
                        predictor=predictor,
                        image_rgb=img_rgb,
                        cams=cams,
                        class_indices=gt_classes,
                        encode_image=encode_flag,
                        **sam_cfg,
                    )

                pseudo_u8 = pseudo_mask.astype(np.uint8)
                if ambiguous_cam_thr < 1.0:
                    max_cam = cams.max(axis=0)
                    ambiguous = (pseudo_u8 == 0) & (max_cam >= ambiguous_cam_thr)
                    pseudo_u8[ambiguous] = IGNORE_INDEX

                save_futures.append(io_pool.submit(_save_uint8_png, pseudo_u8, output_dir / f"{img_id}.png"))
                generated += 1
                pbar.update(1)

            if len(save_futures) >= max(4, batch_sz * 2):
                for fut in save_futures:
                    fut.result()
                save_futures.clear()

        for fut in save_futures:
            fut.result()
    pbar.close()

    if was_training:
        model.train()
    return {"generated": generated, "skipped": skipped, "total": len(image_ids)}


class VOCS2CDataset(Dataset):
    """VOC multi-label dataset with synchronized geometric transforms for SAM segments and CPM masks."""

    def __init__(
        self,
        voc_root: Path,
        labels_df: pd.DataFrame,
        classes: tuple[str, ...] = VOC_CLASSES,
        image_size: int = 448,
        feat_size: int = 28,
        sam_seg_dir: Path | None = None,
        cpm_mask_dir: Path | None = None,
        train: bool = True,
    ) -> None:
        self.voc_root = Path(voc_root)
        self.image_dir = self.voc_root / "JPEGImages"
        self.image_ids = labels_df["image_id"].tolist()
        self.targets = torch.tensor(labels_df[list(classes)].to_numpy(dtype="float32"))
        self.image_size = int(image_size)
        self.feat_size = int(feat_size)
        self.sam_seg_dir = Path(sam_seg_dir) if sam_seg_dir is not None else None
        self.cpm_mask_dir = Path(cpm_mask_dir) if cpm_mask_dir is not None else None
        self.train = bool(train)
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_optional_map(self, directory: Path | None, img_id: str, size_wh: tuple[int, int], fill_value: int) -> Image.Image:
        if directory is not None:
            path = directory / f"{img_id}.png"
            if path.is_file():
                with Image.open(path) as m:
                    return m.convert("L")
        return Image.new("L", size_wh, color=fill_value)

    def __getitem__(self, index: int):
        img_id = self.image_ids[index]
        with Image.open(self.image_dir / f"{img_id}.jpg") as source:
            img = source.convert("RGB")

        W, H = img.size
        seg_pil = self._load_optional_map(self.sam_seg_dir, img_id, (W, H), fill_value=0)
        cpm_pil = self._load_optional_map(self.cpm_mask_dir, img_id, (W, H), fill_value=IGNORE_INDEX)

        if self.train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
            img = TF.resized_crop(img, i, j, h, w, [self.image_size, self.image_size], TF.InterpolationMode.BILINEAR)
            seg_pil = TF.resized_crop(seg_pil, i, j, h, w, [self.feat_size, self.feat_size], TF.InterpolationMode.NEAREST)
            cpm_pil = TF.resized_crop(cpm_pil, i, j, h, w, [self.feat_size, self.feat_size], TF.InterpolationMode.NEAREST)

            if random.random() < 0.5:
                img = TF.hflip(img)
                seg_pil = TF.hflip(seg_pil)
                cpm_pil = TF.hflip(cpm_pil)

            img = self.color_jitter(img)
        else:
            img = TF.resize(img, [self.image_size, self.image_size], TF.InterpolationMode.BILINEAR)
            seg_pil = TF.resize(seg_pil, [self.feat_size, self.feat_size], TF.InterpolationMode.NEAREST)
            cpm_pil = TF.resize(cpm_pil, [self.feat_size, self.feat_size], TF.InterpolationMode.NEAREST)

        img_tensor = TF.to_tensor(img)
        seg_tensor = torch.from_numpy(np.array(seg_pil, dtype=np.int64))
        cpm_tensor = torch.from_numpy(np.array(cpm_pil, dtype=np.int64))
        return img_tensor, self.targets[index], seg_tensor, cpm_tensor


def sam_segment_contrasting_loss(
    proj_feats: torch.Tensor,
    seg_map: torch.Tensor,
    temperature: float = 0.1,
    min_pixels: int = 2,
) -> torch.Tensor:
    """SSC loss: segment-balanced InfoNCE pulling pixel features to their SAM segment prototype."""
    proj_feats = proj_feats.float()
    if seg_map.shape[-2:] != proj_feats.shape[-2:]:
        seg_map = F.interpolate(
            seg_map.unsqueeze(1).float(),
            size=proj_feats.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()

    B, D, _, _ = proj_feats.shape
    losses = []

    for b in range(B):
        seg_b = seg_map[b].reshape(-1)
        valid_mask = seg_b > 0
        if int(valid_mask.sum()) < 2 * min_pixels:
            continue

        seg_valid = seg_b[valid_mask]
        unique_ids, inv_idx, counts = torch.unique(seg_valid, return_inverse=True, return_counts=True)
        keep_seg = counts >= min_pixels
        num_kept = int(keep_seg.sum())
        if num_kept < 2:
            continue

        keep_pixel = keep_seg[inv_idx]
        remap = torch.full((len(unique_ids),), -1, dtype=torch.long, device=seg_map.device)
        remap[keep_seg] = torch.arange(num_kept, device=seg_map.device)
        target_idx = remap[inv_idx[keep_pixel]]

        f_b = proj_feats[b].permute(1, 2, 0).reshape(-1, D)[valid_mask][keep_pixel]
        f_b = F.normalize(f_b, dim=-1)

        proto_sum = torch.zeros((num_kept, D), dtype=f_b.dtype, device=f_b.device)
        proto_sum.index_add_(0, target_idx, f_b)
        seg_counts = torch.bincount(target_idx, minlength=num_kept).clamp_min(1).to(f_b.dtype)
        protos = F.normalize(proto_sum / seg_counts.unsqueeze(1), dim=-1)

        sim_logits = torch.matmul(f_b, protos.T) / float(temperature)
        pix_ce = F.cross_entropy(sim_logits, target_idx, reduction="none")

        seg_ce_sum = torch.zeros((num_kept,), dtype=pix_ce.dtype, device=pix_ce.device)
        seg_ce_sum.index_add_(0, target_idx, pix_ce)
        losses.append((seg_ce_sum / seg_counts).mean())

    if not losses:
        return proj_feats.sum() * 0.0
    return torch.stack(losses).mean()


def cpm_pseudo_mask_loss(
    feat_map: torch.Tensor,
    model: nn.Module,
    cpm_masks: torch.Tensor,
    image_targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """CPM loss: supervise dense spatial classifier logits against SAM-refined pseudo-masks."""
    feat_f32 = feat_map.float()
    if cpm_masks.shape[-2:] != feat_f32.shape[-2:]:
        cpm_masks = F.interpolate(
            cpm_masks.unsqueeze(1).float(),
            size=feat_f32.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()

    head = get_classifier_head(model)
    bias = head.bias.float() if head.bias is not None else None
    fg_logits = F.conv2d(feat_f32, head.weight.float()[:, :, None, None], bias=bias)

    absent = (image_targets <= 0.5)[:, :, None, None]
    fg_logits = fg_logits.masked_fill(absent, -1e4)
    bg_logit = torch.zeros_like(fg_logits[:, :1])
    seg_logits = torch.cat([bg_logit, fg_logits], dim=1)

    safe_targets = cpm_masks.clone()
    fg_pos = (safe_targets > 0) & (safe_targets != ignore_index)
    if fg_pos.any():
        b_idx, y_idx, x_idx = torch.where(fg_pos)
        cls_idx = safe_targets[b_idx, y_idx, x_idx] - 1
        invalid_fg = image_targets[b_idx, cls_idx] <= 0.5
        if invalid_fg.any():
            safe_targets[b_idx[invalid_fg], y_idx[invalid_fg], x_idx[invalid_fg]] = ignore_index

    if not (safe_targets != ignore_index).any():
        return feat_f32.sum() * 0.0

    return F.cross_entropy(seg_logits, safe_targets, ignore_index=ignore_index)


def build_s2c_optimizer(
    model: nn.Module,
    proj_head: nn.Module | None,
    backbone_lr: float,
    head_lr: float | None = None,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer covering backbone, classification head, and optional S2C projection head."""
    from .engine import build_optimizer

    optimizer = build_optimizer(model, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
    if proj_head is not None:
        proj_lr = head_lr if head_lr is not None else backbone_lr
        optimizer.add_param_group({"params": list(proj_head.parameters()), "lr": proj_lr})
    return optimizer


def train_s2c_one_epoch(
    model: nn.Module,
    proj_head: nn.Module | None,
    loader,
    optimizer: torch.optim.Optimizer,
    cls_criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    lambda_ssc: float = 0.2,
    lambda_cpm: float = 0.0,
    ssc_temperature: float = 0.1,
    use_amp: bool = True,
    desc: str = "S2C Train",
) -> dict[str, float]:
    """Run one epoch of S2C training (`L_cls + lambda_ssc * L_ssc + lambda_cpm * L_cpm`)."""
    model.train()
    if proj_head is not None:
        proj_head.train()

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    tot_loss, tot_cls, tot_ssc, tot_cpm, tot_n = 0.0, 0.0, 0.0, 0.0, 0
    progress = tqdm(loader, desc=desc)
    for images, targets, seg_maps, cpm_masks in progress:
        inputs = normalize_batch(images).to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        seg_maps = seg_maps.to(device, non_blocking=True)
        cpm_masks = cpm_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, feat_map = forward_resnet_features(model, inputs)
            loss_cls = cls_criterion(logits, targets)

        loss_ssc = feat_map.new_zeros((), dtype=torch.float32)
        if lambda_ssc > 0.0:
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                proj_feats = proj_head(feat_map) if proj_head is not None else feat_map
            loss_ssc = sam_segment_contrasting_loss(
                proj_feats, seg_maps, temperature=ssc_temperature
            )

        loss_cpm = feat_map.new_zeros((), dtype=torch.float32)
        if lambda_cpm > 0.0:
            loss_cpm = cpm_pseudo_mask_loss(feat_map, model, cpm_masks, targets)

        loss = loss_cls.float() + float(lambda_ssc) * loss_ssc + float(lambda_cpm) * loss_cpm
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = images.size(0)
        tot_n += bs
        tot_loss += float(loss.item()) * bs
        tot_cls += float(loss_cls.item()) * bs
        tot_ssc += float(loss_ssc.item()) * bs
        tot_cpm += float(loss_cpm.item()) * bs
        progress.set_postfix(
            loss=f"{tot_loss / tot_n:.4f}",
            cls=f"{tot_cls / tot_n:.4f}",
            ssc=f"{tot_ssc / tot_n:.4f}",
            cpm=f"{tot_cpm / tot_n:.4f}",
        )

    return {
        "loss": tot_loss / max(tot_n, 1),
        "cls_loss": tot_cls / max(tot_n, 1),
        "ssc_loss": tot_ssc / max(tot_n, 1),
        "cpm_loss": tot_cpm / max(tot_n, 1),
    }
