from __future__ import annotations

import shutil
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .cam import ensure_cam_size

if TYPE_CHECKING:  # segment_anything is only needed at runtime by build_sam_predictor
    from segment_anything import SamPredictor

# model type -> (checkpoint file name, download URL, minimum valid size in bytes)
SAM_MODEL_SPECS = {
    "vit_b": (
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        350_000_000,
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        1_150_000_000,
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        2_400_000_000,
    ),
}


def ensure_sam_checkpoint(model_type: str, target_dir, cache_dirs=()) -> Path:
    """Return the path of a complete SAM checkpoint inside `target_dir`, fetching it if needed.

    Order: reuse a complete local file -> copy from the first complete file found in `cache_dirs`
    (e.g. Google Drive) -> download. Incomplete files (smaller than the expected size) are removed.
    """
    from torchvision.datasets.utils import download_url

    file_name, url, min_bytes = SAM_MODEL_SPECS[model_type]
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / file_name

    if local_path.is_file() and local_path.stat().st_size < min_bytes:
        print(f"{file_name} is incomplete, downloading again...")
        local_path.unlink()
    if local_path.is_file():
        return local_path

    for cache_dir in cache_dirs:
        cached = Path(cache_dir) / file_name
        if cached.is_file() and cached.stat().st_size >= min_bytes:
            print(f"Copying SAM ({model_type}) checkpoint from {cached.parent}...")
            shutil.copy2(cached, local_path)
            return local_path

    print(f"Downloading SAM ({model_type}) checkpoint ({file_name})...")
    download_url(url, str(target_dir), filename=file_name)
    if local_path.stat().st_size < min_bytes:
        raise RuntimeError(f"SAM checkpoint {file_name} was not downloaded completely!")
    return local_path


def build_sam_predictor(model_type: str, checkpoint_path, device) -> "SamPredictor":
    """Load a SAM model from `checkpoint_path`, move it to `device` (eval mode) and wrap it in a SamPredictor."""
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device).eval()
    return SamPredictor(sam)


def get_sam_embedding(predictor: "SamPredictor") -> dict:
    """Snapshot the image embedding of the image last passed to predictor.set_image() (stored on CPU)."""
    return {
        "features_cpu": predictor.features.detach().cpu(),
        "original_size": predictor.original_size,
        "input_size": predictor.input_size,
    }


def set_sam_embedding(predictor: "SamPredictor", embedding: dict, device) -> None:
    """Restore an embedding saved by get_sam_embedding so predictor.predict() works without re-encoding."""
    predictor.features = embedding["features_cpu"].to(device, non_blocking=True)
    predictor.original_size = embedding["original_size"]
    predictor.input_size = embedding["input_size"]
    predictor.is_image_set = True


def _connected_components_2d(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Extract 8-connected components from a 2D binary mask.
    Uses scipy.ndimage.label if available; otherwise falls back to pure-NumPy BFS.
    """
    try:
        from scipy.ndimage import label

        structure = np.ones((3, 3), dtype=np.int32)
        return label(binary_mask, structure=structure)
    except ImportError:
        H, W = binary_mask.shape
        labels = np.zeros((H, W), dtype=np.int32)
        current_label = 0
        for y in range(H):
            for x in range(W):
                if binary_mask[y, x] and labels[y, x] == 0:
                    current_label += 1
                    labels[y, x] = current_label
                    queue = deque([(y, x)])
                    while queue:
                        cy, cx = queue.popleft()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < H and 0 <= nx < W:
                                    if binary_mask[ny, nx] and labels[ny, nx] == 0:
                                        labels[ny, nx] = current_label
                                        queue.append((ny, nx))
        return labels, current_label


def _distance_from_mask(fg_mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (in pixels) from every False pixel to the nearest True pixel in `fg_mask`."""
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(~fg_mask).astype(np.float32)
    except ImportError:
        H, W = fg_mask.shape
        dist = np.full((H, W), np.inf, dtype=np.float32)
        ys, xs = np.where(fg_mask)
        dist[ys, xs] = 0.0
        queue = deque(zip(ys.tolist(), xs.tolist()))
        while queue:
            cy, cx = queue.popleft()
            next_d = dist[cy, cx] + 1.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and next_d < dist[ny, nx]:
                        dist[ny, nx] = next_d
                        queue.append((ny, nx))
        return dist


def _select_negative_points(
    image_uint8: np.ndarray,
    cams: np.ndarray,
    k: int,
    comp_mask: np.ndarray,
    high_core: np.ndarray,
    box: np.ndarray | None,
    pos_pts: list[list[float]],
    cam_threshold: float,
    low_threshold: float,
    bg_threshold: float,
    min_area: int,
    max_neg_points: int,
) -> list[list[float]]:
    """Select confident negative point prompts (label=0) inside `box` for class `k`:
    1. Cross-class negatives: peaks of competing active classes `m != k` inside `box` where `cams[m]`
       clearly dominates `cams[k]` (prevents SAM from merging adjacent/co-occurring objects).
    2. Confident background negative: pixel in `max_c CAM_c <= bg_threshold` buffered away from
       `cams[k] >= low_threshold` AND color-dissimilar from `high_core` (avoids cutting off
       unactivated limbs/parts of the target object).
    """
    if box is None or max_neg_points <= 0:
        return []

    bx0, by0 = int(box[0]), int(box[1])
    bx1, by1 = int(box[2]) + 1, int(box[3]) + 1
    bh, bw = by1 - by0, bx1 - bx0
    if bw < 16 or bh < 16:
        return []

    cam_k_box = cams[k, by0:by1, bx0:bx1]
    comp_box = comp_mask[by0:by1, bx0:bx1]
    fg_zone_box = (cam_k_box >= low_threshold) | comp_box
    if fg_zone_box.all():
        return []

    dist_fg = _distance_from_mask(fg_zone_box)
    neg_pts: list[list[float]] = []
    all_pts: list[list[float]] = [list(p) for p in pos_pts]

    def _far_enough(nx: float, ny: float, min_dist: float) -> bool:
        return all(np.hypot(nx - px, ny - py) >= min_dist for px, py in all_pts)

    # 1. Cross-class negative points from strict high-confidence cores of competing active classes
    if cams.shape[0] > 1:
        cross_candidates = []
        for m in range(cams.shape[0]):
            if m == k:
                continue
            gmax_m = float(cams[m].max())
            if gmax_m < max(cam_threshold, 0.50):
                continue
            cam_m_box = cams[m, by0:by1, bx0:bx1]
            intrude = (
                (cam_m_box >= max(0.55, 0.70 * gmax_m))
                & (~fg_zone_box)
                & (cam_m_box - cam_k_box >= 0.35)
                & (dist_fg >= 6.0)
            )
            if int(intrude.sum()) >= min_area:
                margin = np.where(intrude, cam_m_box - cam_k_box, -1e9)
                ry, rx = np.unravel_index(np.argmax(margin), margin.shape)
                cross_candidates.append((float(margin[ry, rx]), float(bx0 + rx), float(by0 + ry)))

        cross_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, nx, ny in cross_candidates:
            if len(neg_pts) >= max_neg_points:
                break
            if _far_enough(nx, ny, 12.0):
                neg_pts.append([nx, ny])
                all_pts.append([nx, ny])

    # 2. Confident background negative point (anti-overexpansion when CAM covers >=32% of a >=48x48 box)
    fg_ratio = float(fg_zone_box.mean())
    if len(neg_pts) < max_neg_points and bg_threshold > 0.0 and (bw * bh >= 48 * 48) and fg_ratio >= 0.32:
        eff_bg_thr = min(bg_threshold, 0.5 * low_threshold)
        max_any_cam_box = cams[:, by0:by1, bx0:bx1].max(axis=0)
        safe_bg = (max_any_cam_box <= eff_bg_thr) & (dist_fg >= 10.0)

        if int(safe_bg.sum()) >= max(min_area, int(0.08 * cam_k_box.size)):
            core_ref = high_core if high_core.any() else comp_mask
            fg_rgb = image_uint8[core_ref].astype(np.float32).mean(axis=0)
            img_box_f32 = image_uint8[by0:by1, bx0:bx1].astype(np.float32)
            rgb_dist = np.linalg.norm(img_box_f32 - fg_rgb, axis=-1)

            color_safe_bg = safe_bg & (rgb_dist >= 45.0)
            if color_safe_bg.any():
                bg_score = (
                    (1.0 - max_any_cam_box / max(eff_bg_thr, 1e-6))
                    * np.minimum(dist_fg, 36.0)
                    * np.minimum(rgb_dist, 150.0)
                )
                masked_score = np.where(color_safe_bg, bg_score, -1e9)
                ry, rx = np.unravel_index(np.argmax(masked_score), masked_score.shape)
                nx, ny = float(bx0 + rx), float(by0 + ry)
                if _far_enough(nx, ny, 14.0):
                    neg_pts.append([nx, ny])

    return neg_pts


def gen_sam_pseudo_mask(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    cams: np.ndarray,
    class_indices: np.ndarray,
    cam_threshold: float = 0.25,
    low_threshold: float = 0.12,
    bg_threshold: float = 0.05,
    use_neg_points: bool = True,
    max_neg_points: int = 3,
    min_area: int = 25,
    max_components: int = 3,
    box_pad: int = 16,
    encode_image: bool = True,
    w_cam_iou: float = 0.50,
    w_core_recall: float = 0.35,
    w_sam_iou: float = 0.25,
    w_bg_spill: float = 0.45,
    cam_pixel_weight: float = 0.55,
) -> np.ndarray:
    """Generate a pseudo-mask using CAM + SAM (Connected Components + Pos/Neg Prompts + Multi-mask):
    1. Align `cams` to `image_rgb` resolution `(H, W)` and split active CAM regions into components.
    2. Extract Positive Points (Peak + Core Centroid), Confident Negative Points (competing-class peaks
       + buffered color-distinct background), and a Padded Bounding Box per component.
    3. Run SAM with `multimask_output=True` and select the candidate mask best aligned with the CAM.
    """
    if image_rgb.max() <= 1.0:
        image_uint8 = (np.clip(image_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        image_uint8 = image_rgb.astype(np.uint8)
    if encode_image:
        predictor.set_image(image_uint8)

    H, W = image_uint8.shape[:2]
    cams = ensure_cam_size(cams, (H, W))
    if cams.ndim == 2:
        cams = np.expand_dims(cams, axis=0)
    class_indices = np.asarray(class_indices, dtype=np.int64)

    pseudo_mask = np.zeros((H, W), dtype=np.int64)
    best_scores = np.zeros((H, W), dtype=np.float32)

    for k in range(cams.shape[0]):
        cam_k = cams[k]
        voc_class_id = int(class_indices[k]) + 1
        global_max = float(cam_k.max())
        if global_max < 1e-4:
            continue

        if cams.shape[0] > 1:
            other_max = np.max(np.delete(cams, k, axis=0), axis=0)
            spill_mask = (cam_k < 0.08) | ((other_max >= cam_threshold) & (cam_k < low_threshold))
        else:
            spill_mask = cam_k < 0.08

        # 1. Split active region (cam_k >= cam_threshold) into connected components
        active_region = cam_k >= cam_threshold
        labeled_map, num_comps = _connected_components_2d(active_region)

        valid_comps = []
        for comp_id in range(1, num_comps + 1):
            comp_mask = labeled_map == comp_id
            area = int(comp_mask.sum())
            comp_max = float(cam_k[comp_mask].max())
            # Keep components with sufficient area and peak activation relative to the global maximum
            if area >= min_area and comp_max >= max(cam_threshold, 0.55 * global_max):
                valid_comps.append((area, comp_max, comp_mask))

        # Sort descending by peak activation and area; keep at most max_components
        valid_comps.sort(key=lambda item: (item[1], item[0]), reverse=True)
        valid_comps = valid_comps[:max_components]

        # Fallback if no component passes the filter: use the global peak region of cam_k
        if len(valid_comps) == 0:
            fallback_mask = cam_k >= max(low_threshold, 0.5 * global_max)
            if not fallback_mask.any():
                continue
            valid_comps = [(int(fallback_mask.sum()), global_max, fallback_mask)]

        class_mask_k = np.zeros((H, W), dtype=bool)

        for _, comp_max, comp_mask in valid_comps:
            # 2A. Find the local peak point inside the component -> Positive Point 1
            masked_cam = np.where(comp_mask, cam_k, -1.0)
            peak_y, peak_x = np.unravel_index(np.argmax(masked_cam), cam_k.shape)
            pts: list[list[float]] = [[float(peak_x), float(peak_y)]]
            lbls: list[int] = [1]

            # 2B. Find the centroid of the high-confidence core (cam >= 0.7 * comp_max) -> Positive Point 2
            high_core = comp_mask & (cam_k >= 0.70 * comp_max)
            core_ys, core_xs = np.where(high_core)
            if len(core_xs) >= 25:
                cy, cx = float(core_ys.mean()), float(core_xs.mean())
                cy_i = int(np.clip(round(cy), 0, H - 1))
                cx_i = int(np.clip(round(cx), 0, W - 1))
                # Snap centroid into high_core if concave geometry placed it outside the core
                if not high_core[cy_i, cx_i]:
                    nearest_idx = int(np.argmin((core_xs - cx) ** 2 + (core_ys - cy) ** 2))
                    cy, cx = float(core_ys[nearest_idx]), float(core_xs[nearest_idx])
                if np.hypot(cx - peak_x, cy - peak_y) >= 8.0:
                    pts.append([cx, cy])
                    lbls.append(1)

            # 2C. Build a padded bounding box around the component to avoid cutting off object parts
            ys, xs = np.where(comp_mask)
            if len(xs) > 0:
                x0 = max(0, int(xs.min()) - box_pad)
                y0 = max(0, int(ys.min()) - box_pad)
                x1 = min(W - 1, int(xs.max()) + box_pad)
                y1 = min(H - 1, int(ys.max()) + box_pad)
                box = np.array([x0, y0, x1, y1], dtype=np.float32)
            else:
                box = None

            # 2D. Extract Confident Negative Points (competing-class peaks + safe background)
            if use_neg_points and max_neg_points > 0:
                neg_pts = _select_negative_points(
                    image_uint8=image_uint8,
                    cams=cams,
                    k=k,
                    comp_mask=comp_mask,
                    high_core=high_core,
                    box=box,
                    pos_pts=pts,
                    cam_threshold=cam_threshold,
                    low_threshold=low_threshold,
                    bg_threshold=bg_threshold,
                    min_area=min_area,
                    max_neg_points=max_neg_points,
                )
                for nx, ny in neg_pts:
                    pts.append([nx, ny])
                    lbls.append(0)

            point_coords = np.array(pts, dtype=np.float32)
            point_labels = np.array(lbls, dtype=np.int32)

            # 3. Enable multimask_output=True so SAM proposes 3 mask levels (Part / Whole / Context)
            masks, iou_preds, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

            # Expanded reference CAM region within the box to evaluate whole-object masks
            ref_region = cam_k >= low_threshold
            if box is not None:
                box_mask = np.zeros((H, W), dtype=bool)
                box_mask[int(box[1]) : int(box[3]) + 1, int(box[0]) : int(box[2]) + 1] = True
                ref_region = ref_region & box_mask

            best_m_idx = 0
            best_m_score = -1e9
            for m_idx in range(masks.shape[0]):
                cand_mask = masks[m_idx]
                cand_area = float(cand_mask.sum())
                if cand_area < 1.0:
                    continue

                # IoU alignment with the expanded CAM region of the component
                inter = float((cand_mask & ref_region).sum())
                union = float((cand_mask | ref_region).sum()) + 1e-6
                cam_iou = inter / union

                # Core recall (ensures the candidate mask covers the high-confidence CAM core)
                core_recall = float((cand_mask & comp_mask).sum()) / (float(comp_mask.sum()) + 1e-6)

                # Fraction of mask spilling into confident background or competing classes
                bg_spill = float((cand_mask & spill_mask).sum()) / cand_area

                # Composite score for selecting the optimal mask candidate
                total_score = (
                    w_cam_iou * cam_iou
                    + w_core_recall * core_recall
                    + w_sam_iou * float(iou_preds[m_idx])
                    - w_bg_spill * bg_spill
                )
                if total_score > best_m_score:
                    best_m_score = total_score
                    best_m_idx = m_idx

            class_mask_k |= masks[best_m_idx]

        # 4. Assign class label to pseudo_mask using combined pixel CAM and mean mask confidence
        if class_mask_k.any():
            mean_mask_cam = float(cam_k[class_mask_k].mean())
            combined_score = cam_pixel_weight * cam_k + (1.0 - cam_pixel_weight) * mean_mask_cam
            update_region = class_mask_k & (combined_score >= best_scores)
            pseudo_mask[update_region] = voc_class_id
            best_scores[update_region] = combined_score[update_region]

    return pseudo_mask
