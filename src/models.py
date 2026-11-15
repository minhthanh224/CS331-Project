import torch.nn as nn
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large, ConvNeXt_Large_Weights,
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)

# ImageNet-pretrained classifiers. Every entry ends with a GAP -> Linear (`fc` for ResNet,
# `classifier[2]` for ConvNeXt) so the same traditional-CAM math applies to both families.
MODEL_REGISTRY = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
    "convnext_base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
    "convnext_large": (convnext_large, ConvNeXt_Large_Weights.DEFAULT),
}

CONVNEXT_MODEL_NAMES = frozenset(name for name in MODEL_REGISTRY if name.startswith("convnext"))


def is_convnext(model: nn.Module) -> bool:
    """True when `model` is a torchvision ConvNeXt (features + avgpool + classifier head)."""
    return hasattr(model, "features") and hasattr(model, "classifier") and not hasattr(model, "fc")


def get_classifier_head(model: nn.Module) -> nn.Linear:
    """Return the final Linear layer of a ResNet (`fc`) or ConvNeXt (`classifier[2]`) model."""
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        for module in reversed(model.classifier):
            if isinstance(module, nn.Linear):
                return module
    raise AttributeError(f"{type(model).__name__} has no fc/classifier Linear head")


def default_layercam_layers(model: nn.Module) -> tuple[str, ...]:
    """Deepest convolutional stage, named for `getattr`/LayerCAM hooks.

    - ResNet: `("layer4",)`
    - ConvNeXt: `("features.7",)` (last stage). Use `("features.5", "features.7")` or
      `("features.3", "features.5", "features.7")` to fuse higher-resolution stages.
    """
    if is_convnext(model):
        return (f"features.{len(model.features) - 1}",)
    return ("layer4",)


def _set_convnext_last_stride1(model: nn.Module) -> None:
    """Turn the last ConvNeXt downsample (2x2, stride 2) into stride 1 for a ~2x CAM map."""
    downsample = model.features[-2]
    conv = downsample[1] if isinstance(downsample, nn.Sequential) else downsample
    if isinstance(conv, nn.Conv2d) and conv.stride == (2, 2):
        conv.stride = (1, 1)


def build_classifier(
    model_name: str = "resnet18",
    num_classes: int = 20,
    layer4_stride1: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    """Build a classifier with a fresh `num_classes` head (ImageNet-pretrained by default).

    `layer4_stride1` applies to ResNet's `layer4[0]` and to ConvNeXt's last downsample stage
    (2x higher-resolution feature maps, useful for CAM). Pretrained weights are unaffected because
    only the stride changes.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"model_name must be one of {list(MODEL_REGISTRY)}, got: {model_name}")

    model_fn, weights = MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights if pretrained else None)

    if model_name in CONVNEXT_MODEL_NAMES:
        if layer4_stride1:
            _set_convnext_last_stride1(model)
        head = get_classifier_head(model)
        for i in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[i], nn.Linear):
                model.classifier[i] = nn.Linear(head.in_features, num_classes)
                break
    else:
        if layer4_stride1:
            block = model.layer4[0]
            if hasattr(block, "conv2") and block.conv2.stride == (2, 2):
                block.conv2.stride = (1, 1)  # Bottleneck (ResNet-50 / ResNet-101)
            elif hasattr(block, "conv1") and block.conv1.stride == (2, 2):
                block.conv1.stride = (1, 1)  # BasicBlock (ResNet-18)
            if block.downsample is not None and block.downsample[0].stride == (2, 2):
                block.downsample[0].stride = (1, 1)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    return model


def _infer_num_classes_from_state(state: dict) -> int | None:
    """Infer output class count from a classifier state_dict's final Linear weight."""
    if "fc.weight" in state:
        return int(state["fc.weight"].shape[0])
    classifier_weights = [
        (k, v) for k, v in state.items()
        if k.startswith("classifier.") and k.endswith(".weight") and v.ndim == 2
    ]
    if classifier_weights:
        classifier_weights.sort(key=lambda item: item[0])
        return int(classifier_weights[-1][1].shape[0])
    return None


def load_classifier(
    model_name: str,
    checkpoint_path,
    device="cpu",
    num_classes: int | None = None,
    layer4_stride1: bool | None = None,
) -> nn.Module:
    """Build a classifier, load a checkpoint into it and return it in eval mode on `device`.

    Accepts both a plain state_dict file and a training payload with a "model_state_dict" key.
    When `num_classes is None`, infers class count from payload metadata or head weights (default 20).
    When `layer4_stride1 is None`, automatically reads `config["layer4_stride1"]` from payload
    checkpoints if available (defaulting to False for plain state_dict files).
    """
    import torch

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        if num_classes is None:
            if "num_classes" in state and state["num_classes"] is not None:
                num_classes = int(state["num_classes"])
            elif isinstance(state.get("classes"), (list, tuple)):
                num_classes = len(state["classes"])
        if layer4_stride1 is None:
            cfg = state.get("config")
            if isinstance(cfg, dict) and "layer4_stride1" in cfg:
                layer4_stride1 = bool(cfg["layer4_stride1"])
        state = state["model_state_dict"]

    if num_classes is None:
        inferred = _infer_num_classes_from_state(state) if isinstance(state, dict) else None
        num_classes = inferred if inferred is not None else 20

    if layer4_stride1 is None:
        layer4_stride1 = False

    model = build_classifier(model_name, num_classes=num_classes, layer4_stride1=layer4_stride1, pretrained=False)
    model.load_state_dict(state)
    return model.to(device).eval()
