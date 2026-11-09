import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)

MODEL_REGISTRY = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
}

def build_classifier(
    model_name: str = "resnet18",
    num_classes: int = 20,
    layer4_stride1: bool = False,
) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"model_name must be one of {list(MODEL_REGISTRY)}, got: {model_name}")
    
    model_fn, weights = MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights)
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


def load_classifier(
    model_name: str,
    checkpoint_path,
    device="cpu",
    num_classes: int = 20,
    layer4_stride1: bool = False,
) -> nn.Module:
    """Build a classifier, load a checkpoint into it and return it in eval mode on `device`.

    Accepts both a plain state_dict file and a training payload with a "model_state_dict" key.
    """
    import torch

    model = build_classifier(model_name, num_classes=num_classes, layer4_stride1=layer4_stride1)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()
