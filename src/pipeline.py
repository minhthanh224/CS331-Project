"""Generic evaluation pipeline: CAM generation + pseudo-mask methods + segmentation metrics.

A *mask function* has the signature  fn(img_rgb, cams, active_classes) -> (H, W) int64 mask.
A *cam function*  has the signature  fn(img_tensor, active_classes[, target_size]) -> (K, H, W) float CAMs.

Build them with the make_* factories below, describe each backbone with a BackbonePipeline and let
evaluate_segmentation() run the shared loop at native Pascal VOC image resolution (`image_size=None`
by default) or a custom target resolution.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from .cam import compute_cams, ensure_cam_size, gen_cam_pseudo_mask, select_active_classes
from .constants import IGNORE_INDEX, NUM_SEG_CLASSES
from .dataset import VOCClassificationDataset, load_eval_sample
from .dense_CRF import get_dense_crf_mask
from .metrics import SegmentationEvaluator
from .sam_utils import gen_sam_pseudo_mask

MaskFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
CamFn = Callable[..., np.ndarray]


def _sample_target_size(sample: dict) -> tuple[int, int] | None:
    if "gt_mask" in sample and sample["gt_mask"] is not None:
        return tuple(sample["gt_mask"].shape[:2])
    if "img_rgb" in sample and sample["img_rgb"] is not None:
        return tuple(sample["img_rgb"].shape[:2])
    if "img_u8" in sample and sample["img_u8"] is not None:
        return tuple(sample["img_u8"].shape[:2])
    return None


def _invoke_cam_fn(
    cam_fn: CamFn,
    img_tensor: torch.Tensor,
    active_classes: np.ndarray,
    target_size: tuple[int, int] | None,
) -> np.ndarray:
    params = inspect.signature(cam_fn).parameters
    accepts_target = "target_size" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_target:
        cams = cam_fn(img_tensor, active_classes, target_size=target_size)
    else:
        cams = cam_fn(img_tensor, active_classes)
    return ensure_cam_size(cams, target_size)


# ---------------------------------------------------------------------------------------------
# Factories for cam functions and mask functions
# ---------------------------------------------------------------------------------------------
def make_cam_fn(
    model: torch.nn.Module,
    method: str = "cam",
    scales: tuple[float, ...] = (1.0,),
    hflip: bool = False,
    layers: tuple[str, ...] = ("layer4",),
    shallow_gamma: float = 2.0,
) -> CamFn:
    """CAM / LayerCAM generator bound to a model and its TTA settings."""
    def cam_fn(
        img_tensor: torch.Tensor,
        active_classes,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        return compute_cams(
            model, img_tensor, active_classes, method=method,
            target_size=target_size, scales=scales, hflip=hflip,
            layers=layers, shallow_gamma=shallow_gamma,
        )
    return cam_fn


def make_cam_mask_fn(cam_threshold: float) -> MaskFn:
    """Vanilla pseudo-mask: pixels with CAM >= threshold take the strongest class, the rest is background."""
    def mask_fn(img_rgb, cams, active_classes):
        target_hw = img_rgb.shape[:2] if img_rgb is not None else None
        return gen_cam_pseudo_mask(cams, active_classes, cam_threshold=cam_threshold, target_size=target_hw)
    return mask_fn


def make_crf_mask_fn(**crf_params) -> MaskFn:
    """CAM + DenseCRF; crf_params go to get_dense_crf_mask (alpha, compat_bi, sxy_bi, srgb_bi, n_iters, ...)."""
    def mask_fn(img_rgb, cams, active_classes):
        return get_dense_crf_mask(image_rgb=img_rgb, cams=cams, class_indices=active_classes, **crf_params)
    return mask_fn


def make_sam_mask_fn(predictor, encode_image: bool = True, **sam_params) -> MaskFn:
    """CAM + SAM; sam_params go to gen_sam_pseudo_mask (cam_threshold, low_threshold, bg_threshold,
    use_neg_points, box_pad, w_*, ...).

    Use encode_image=False when the image embedding was already set (e.g. shared across backbones via
    evaluate_segmentation(prepare_fn=...)).
    """
    def mask_fn(img_rgb, cams, active_classes):
        return gen_sam_pseudo_mask(
            predictor=predictor, image_rgb=img_rgb, cams=cams, class_indices=active_classes,
            encode_image=encode_image, **sam_params,
        )
    return mask_fn


# ---------------------------------------------------------------------------------------------
# CAM caching (used when tuning hyper-parameters on a fixed set of images)
# ---------------------------------------------------------------------------------------------
def cache_cams(
    samples: list[dict],
    probs: np.ndarray,
    cls_threshold: float,
    cam_fns: dict[str, CamFn],
    desc: str | None = None,
) -> dict[str, list[dict]]:
    """Compute every cam function once per sample (aligned to each sample's evaluation resolution).

    samples: dicts from load_eval_sample (need "img_tensor" and optionally "gt_mask"/"img_rgb");
    probs: (len(samples), num_classes) probabilities aligned with `samples`.
    Returns {cam_name: [{"active_classes": ..., "cams": ...}, ...]} aligned with `samples`.
    """
    cache = {name: [] for name in cam_fns}
    iterator = tqdm(range(len(samples)), desc=desc) if desc else range(len(samples))
    for i in iterator:
        sample = samples[i]
        active = select_active_classes(probs[i], cls_threshold)
        target_size = _sample_target_size(sample)
        for name, cam_fn in cam_fns.items():
            cams = _invoke_cam_fn(cam_fn, sample["img_tensor"], active, target_size)
            cache[name].append({"active_classes": active, "cams": cams})
    return cache


# ---------------------------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------------------------
@dataclass
class BackbonePipeline:
    """Everything evaluate_segmentation needs to know about one backbone.

    probs: (N, num_classes) classifier probabilities aligned with the evaluated dataset.
    cls_threshold: probability threshold selecting the active classes of each image.
    cam_fns: {cam_name: cam function}, e.g. {"cam": ..., "layercam": ...}.
    methods: {method_label: (cam_name, mask function)}, e.g. {"CAM + SAM": ("cam", sam_fn)}.
    """
    probs: np.ndarray
    cls_threshold: float
    cam_fns: dict[str, CamFn]
    methods: dict[str, tuple[str, MaskFn]] = field(default_factory=dict)


def predict_sample_masks(sample: dict, pipe: BackbonePipeline, prob_vec: np.ndarray | None = None) -> dict:
    """Run all CAM functions and mask methods of `pipe` on a single sample dict from `load_eval_sample`.

    CAMs are generated directly at `sample["gt_mask"].shape` (native image resolution when
    `load_eval_sample` is called with `image_size=None`).
    Returns {"active_classes": np.ndarray, "cams": {cam_name: np.ndarray}, "masks": {method_label: np.ndarray}}.
    """
    if prob_vec is None:
        prob_vec = pipe.probs[sample["idx"]]
    active = select_active_classes(prob_vec, pipe.cls_threshold)
    target_size = _sample_target_size(sample)
    cams_by_type: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for label, (cam_name, mask_fn) in pipe.methods.items():
        if cam_name not in cams_by_type:
            cams_by_type[cam_name] = _invoke_cam_fn(
                pipe.cam_fns[cam_name], sample["img_tensor"], active, target_size
            )
        masks[label] = mask_fn(sample["img_rgb"], cams_by_type[cam_name], active)
    return {
        "active_classes": active,
        "cams": cams_by_type,
        "masks": masks,
    }


def build_optuna_pipeline(
    model: torch.nn.Module,
    probs: np.ndarray,
    best_params: dict,
    predictor=None,
    encode_image: bool = False,
    scales: tuple[float, ...] | None = None,
    hflip: bool | None = None,
    layercam_layers: tuple[str, ...] | None = None,
    shallow_gamma: float | None = None,
) -> BackbonePipeline:
    """Build a BackbonePipeline from an Optuna `best_params` dict (supports up to 5 methods:
    Vanilla CAM, CAM + DenseCRF, CAM + SAM, LayerCAM, LayerCAM + SAM).
    """
    scales = tuple(scales if scales is not None else best_params.get("cam_scales", (1.0,)))
    hflip = bool(hflip if hflip is not None else best_params.get("cam_hflip", False))
    layercam_layers = tuple(
        layercam_layers if layercam_layers is not None else best_params.get("layercam_layers", ("layer4",))
    )
    shallow_gamma = float(
        shallow_gamma if shallow_gamma is not None else best_params.get("layercam_shallow_gamma", 2.0)
    )

    cam_fns: dict[str, CamFn] = {
        "cam": make_cam_fn(model, method="cam", scales=scales, hflip=hflip),
    }
    if "best_layercam_params" in best_params or "best_layercam_sam_params" in best_params:
        cam_fns["layercam"] = make_cam_fn(
            model, method="layercam", scales=scales, hflip=hflip,
            layers=layercam_layers, shallow_gamma=shallow_gamma,
        )

    methods: dict[str, tuple[str, MaskFn]] = {}
    if "best_cam_params" in best_params:
        methods["Vanilla CAM"] = ("cam", make_cam_mask_fn(best_params["best_cam_params"]["cam_threshold"]))
    if "best_crf_params" in best_params:
        methods["CAM + DenseCRF"] = ("cam", make_crf_mask_fn(**best_params["best_crf_params"]))
    if predictor is not None and "best_sam_params" in best_params:
        methods["CAM + SAM"] = (
            "cam",
            make_sam_mask_fn(predictor, encode_image=encode_image, **best_params["best_sam_params"]),
        )
    if "best_layercam_params" in best_params:
        methods["LayerCAM"] = (
            "layercam",
            make_cam_mask_fn(best_params["best_layercam_params"]["cam_threshold"]),
        )
    if predictor is not None and "best_layercam_sam_params" in best_params:
        methods["LayerCAM + SAM"] = (
            "layercam",
            make_sam_mask_fn(predictor, encode_image=encode_image, **best_params["best_layercam_sam_params"]),
        )

    return BackbonePipeline(
        probs=probs,
        cls_threshold=float(best_params.get("best_cls_threshold", 0.5)),
        cam_fns=cam_fns,
        methods=methods,
    )


def evaluate_segmentation(
    dataset: VOCClassificationDataset,
    pipelines: dict[str, BackbonePipeline],
    image_size: int | tuple[int, int] | None = None,
    prepare_fn: Callable[[np.ndarray], None] | None = None,
    indices=None,
    num_classes: int = NUM_SEG_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    desc: str = "Segmentation evaluation",
) -> dict[str, dict[str, dict]]:
    """Evaluate every method of every backbone on the dataset in a single pass over the images.

    - `image_size=None` (default): standard Pascal VOC benchmark protocol — evaluates masks against
      native-resolution Ground Truth `(H_orig, W_orig)` while feeding `dataset[idx]` (e.g. 448x448)
      into the CNN for CAM extraction.
    - `prepare_fn(img_u8)` is called once per image before any mask function, so work shared by all
      backbones (such as SAM image encoding) is done only once at native resolution.
    Returns {backbone: {method_label: metrics_dict}} with metrics from SegmentationEvaluator.compute().
    """
    evaluators = {
        name: {label: SegmentationEvaluator(num_classes=num_classes, ignore_index=ignore_index) for label in p.methods}
        for name, p in pipelines.items()
    }
    indices = range(len(dataset)) if indices is None else indices

    for idx in tqdm(indices, desc=desc):
        sample = load_eval_sample(dataset, idx, image_size=image_size)
        if prepare_fn is not None:
            prepare_fn(sample["img_u8"])

        for name, pipe in pipelines.items():
            out = predict_sample_masks(sample, pipe, pipe.probs[idx])
            for label, mask in out["masks"].items():
                evaluators[name][label].update(mask, sample["gt_mask"])

    return {name: {label: ev.compute() for label, ev in per_method.items()} for name, per_method in evaluators.items()}
