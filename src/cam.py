import torch.nn as nn
import torchvision.models
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

def gen_cam_resnet(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_indices,
    target_size=None,
    scales: tuple[float, ...] = (1.0,),
    hflip: bool = False,
):
    """
    Generate Class Activation Maps (CAM) for a ResNet model.
    Supports optional Multi-scale + Horizontal Flip Test-Time Augmentation (TTA).

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
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    normalize = transforms.Normalize(mean=mean, std=std)
    image = normalize(input_tensor)

    # Default output resolution to the input image's (H, W) if target_size is not specified
    orig_h, orig_w = image.shape[-2:]
    if target_size is None:
        target_size = (orig_h, orig_w)

    with torch.no_grad():
        # Get classifier weights for the target classes: (K, C) or (B, C)
        class_weights = model.fc.weight[class_indices]
        # Reshape weights to (K, C, 1, 1) or (B, C, 1, 1) for spatial broadcasting
        weights = class_weights[:, :, None, None]

        accum_cam = None
        for s in scales:
            if s == 1.0:
                scaled_img = image
            else:
                sh = max(32, int(round(orig_h * s)))
                sw = max(32, int(round(orig_w * s)))
                scaled_img = F.interpolate(image, size=(sh, sw), mode="bilinear", align_corners=False)

            # Forward original orientation at scale s
            fmap = feature_extractor(scaled_img)
            cam_s = torch.relu((fmap * weights).sum(dim=1, keepdim=True))
            cam_s = F.interpolate(cam_s, size=target_size, mode="bilinear", align_corners=False).squeeze(1)
            cam_s = cam_s / cam_s.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            accum_cam = cam_s if accum_cam is None else (accum_cam + cam_s)

            # Optional Horizontal Flip TTA at scale s
            if hflip:
                flipped_img = torch.flip(scaled_img, dims=[-1])
                fmap_f = feature_extractor(flipped_img)
                cam_f = torch.relu((fmap_f * weights).sum(dim=1, keepdim=True))
                cam_f = torch.flip(cam_f, dims=[-1])
                cam_f = F.interpolate(cam_f, size=target_size, mode="bilinear", align_corners=False).squeeze(1)
                cam_f = cam_f / cam_f.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                accum_cam = accum_cam + cam_f

        # Min-max normalize the aggregated multi-scale heatmap independently to [0, 1]
        cam = accum_cam / accum_cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

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
