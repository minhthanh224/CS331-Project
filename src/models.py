import torch.nn as nn
from torchvision.models import (resnet18, ResNet18_Weights, resnet50, ResNet50_Weights)

def build_classifier(model_name: str = "resnet18", num_classes: int = 20) -> nn.Module:
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"model_name must be 'resnet18' or 'resnet50', got: {model_name}")
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    
    return model
