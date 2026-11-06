import torch.nn as nn
import torchvision.models
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

def gen_cam_resnet(model: nn.Module, input_tensor: torch.Tensor , class_indices, target_size = None):
    """
    Generate Class Activation Maps (CAM) for a ResNet model.

    Supports two broadcasting modes:
    1. Multi-class on a single image:
       - input_tensor: (3, H, W) or (1, 3, H, W)
       - class_indices: (K,) indices of K target classes in the image
       - Returns: (K, H, W) CAM heatmaps for all K classes in a single forward pass.
    2. Single-class per image on a batch:
       - input_tensor: (B, 3, H, W)
       - class_indices: (B,) one target class index per image
       - Returns: (B, H, W) CAM heatmaps, one for each image in the batch.
    """
    # Remove the last two layers (avgpool and fc) to extract convolutional feature maps
    feature_extractor = nn.Sequential(*list(model.children())[:-2]).eval()

    # ImageNet normalization parameters required by pretrained ResNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # If a single image (3, H, W) is passed, add a batch dimension -> (1, 3, H, W)
    if(input_tensor).dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    normalize = transforms.Normalize(mean=mean, std=std)

    image = normalize(input_tensor)
    with torch.no_grad():
        # Extract spatial feature maps from the last conv block: (B, C, h, w), e.g., (B, 512, 8, 8)
        feature_maps = feature_extractor.forward(image)

        # Get classifier weights for the target classes: (K, C) or (B, C)
        class_weights = model.fc.weight[class_indices]

        # Reshape weights to (K, C, 1, 1) or (B, C, 1, 1) for spatial broadcasting
        weights = class_weights[:, :, None, None]

        # Compute weighted sum across the channel dimension (dim=1):
        # - If B=1 and K classes: (1, C, h, w) * (K, C, 1, 1) -> summed to (K, 1, h, w)
        # - If B images and B classes: (B, C, h, w) * (B, C, 1, 1) -> summed to (B, 1, h, w)
        raw_cam = (feature_maps * weights).sum(dim=1, keepdim=True)

        # Keep only positive activations that positively contribute to the target class
        cam = torch.relu(raw_cam)

    # Default output resolution to the input image's (H, W) if target_size is not specified
    if target_size is None:
        target_size = image.shape[-2:]

    # Upsample low-resolution CAM (h, w) back to target_size (H, W) and remove channel dim -> (K, H, W) or (B, H, W)
    cam = F.interpolate(
        cam, 
        size=target_size, 
        mode="bilinear", 
        align_corners=False
    ).squeeze(1)
    
    # Min-max normalize each heatmap independently to [0, 1] (clamp_min prevents division by zero)
    cam = cam / cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    return cam

# def get_bg_score(cams, alpha)

def gen_cam_pseudo_mask(cams, class_indices, cam_threshhold: float = 0.25) -> np.ndarray:
    if torch.is_tensor(cams):
        cams = cams.detach().cpu().numpy()
    else:
        cams = np.asarray(cams)
    if torch.is_tensor(class_indices):
        class_indices = class_indices.detach().cpu().numpy()
    else:
        class_indices = np.asarray(class_indices)
    if cams.ndim == 2:
        cams = np.expand_dims(cams, axis=0)

    voc_class_ids = class_indices.astype(np.int64) + 1 #Add 1 to class indices to count for background class
    max_cam = cams.max(axis=0) #Get max cam heatmap value
    best_k = cams.argmax(axis = 0) #Get the k index of the layer that has the greatest cam

    pseudo_mask = np.where(max_cam >= cam_threshhold, voc_class_ids[best_k], 0)
    return pseudo_mask.astype(np.int64)
