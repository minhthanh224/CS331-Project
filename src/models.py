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

def build_classifier(model_name: str = "resnet18", num_classes: int = 20) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"model_name must be one of {list(MODEL_REGISTRY)}, got: {model_name}")
    
    model_fn, weights = MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    return model
