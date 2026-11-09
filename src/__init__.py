"""WSSS (Weakly Supervised Semantic Segmentation) toolkit: CAM, LayerCAM, DenseCRF, and SAM."""

from .cam import (
    compute_cams,
    gen_cam_pseudo_mask,
    gen_cam_resnet,
    gen_layercam_resnet,
    select_active_classes,
)
from .config import TrainConfig
from .constants import (
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_SEG_CLASSES,
    NUM_VOC_CLASSES,
    SUPPORTED_BACKBONES,
    VOC_CLASSES,
)
from .dataset import (
    VOCClassificationDataset,
    build_transforms,
    load_eval_sample,
    load_eval_samples,
    load_gt_mask,
    make_loader,
    read_split_ids,
)
from .dense_CRF import dense_crf_pytorch, get_dense_crf_mask
from .engine import (
    build_optimizer,
    normalize_batch,
    poly_scheduler,
    predict_probs,
    train_one_epoch,
)
from .labels import (
    build_balanced_tune_table,
    build_class_to_index,
    build_voc_label_table,
    encode_multihot,
    read_voc_labels,
)
from .logger import (
    cls_summary_row,
    log_eval_results,
    print_cls_metrics,
    print_seg_metrics,
    seg_summary_row,
)
from .metrics import SegmentationEvaluator, compute_classification_metrics
from .models import MODEL_REGISTRY, build_classifier, load_classifier
from .pipeline import (
    BackbonePipeline,
    build_optuna_pipeline,
    cache_cams,
    evaluate_segmentation,
    make_cam_fn,
    make_cam_mask_fn,
    make_crf_mask_fn,
    make_sam_mask_fn,
    predict_sample_masks,
)
from .sam_utils import (
    SAM_MODEL_SPECS,
    build_sam_predictor,
    ensure_sam_checkpoint,
    gen_sam_pseudo_mask,
    get_sam_embedding,
    set_sam_embedding,
)
from .utils import (
    add_project_root_to_path,
    find_project_root,
    get_device,
    in_colab,
    set_seed,
    to_uint8_image,
)
from .visualize import (
    draw_cell,
    heatmap_cell,
    image_cell,
    mask_cell,
    overlay_cell,
    plot_grid,
)

__all__ = [
    # constants
    "VOC_CLASSES",
    "NUM_VOC_CLASSES",
    "NUM_SEG_CLASSES",
    "IGNORE_INDEX",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SUPPORTED_BACKBONES",
    # config & utils
    "TrainConfig",
    "set_seed",
    "get_device",
    "in_colab",
    "find_project_root",
    "add_project_root_to_path",
    "to_uint8_image",
    # labels & dataset
    "encode_multihot",
    "build_class_to_index",
    "read_voc_labels",
    "build_voc_label_table",
    "build_balanced_tune_table",
    "VOCClassificationDataset",
    "read_split_ids",
    "build_transforms",
    "make_loader",
    "load_gt_mask",
    "load_eval_sample",
    "load_eval_samples",
    # models & engine
    "MODEL_REGISTRY",
    "build_classifier",
    "load_classifier",
    "normalize_batch",
    "build_optimizer",
    "poly_scheduler",
    "train_one_epoch",
    "predict_probs",
    # CAM / DenseCRF / SAM
    "gen_cam_resnet",
    "gen_layercam_resnet",
    "gen_cam_pseudo_mask",
    "select_active_classes",
    "compute_cams",
    "dense_crf_pytorch",
    "get_dense_crf_mask",
    "SAM_MODEL_SPECS",
    "ensure_sam_checkpoint",
    "build_sam_predictor",
    "get_sam_embedding",
    "set_sam_embedding",
    "gen_sam_pseudo_mask",
    # pipeline, metrics, logger, visualize
    "BackbonePipeline",
    "make_cam_fn",
    "make_cam_mask_fn",
    "make_crf_mask_fn",
    "make_sam_mask_fn",
    "cache_cams",
    "predict_sample_masks",
    "build_optuna_pipeline",
    "evaluate_segmentation",
    "compute_classification_metrics",
    "SegmentationEvaluator",
    "log_eval_results",
    "seg_summary_row",
    "cls_summary_row",
    "print_seg_metrics",
    "print_cls_metrics",
    "image_cell",
    "mask_cell",
    "heatmap_cell",
    "overlay_cell",
    "draw_cell",
    "plot_grid",
]
