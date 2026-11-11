import torch.nn as nn
import torchvision.models
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

from .constants import IMAGENET_MEAN, IMAGENET_STD

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

    # If a single image (3, H, W) is passed, add a batch dimension -> (1, 3, H, W)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # ImageNet normalization parameters required by pretrained ResNet
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
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


def _layercam_single_pass(
    model: nn.Module,
    image: torch.Tensor,
    class_indices: torch.Tensor,
    layers: tuple[str, ...],
    target_size,
    shallow_gamma: float,
) -> torch.Tensor:
    """Run one forward/backward pass and return the fused LayerCAM: (N, H, W) in [0, 1]."""
    # Multi-class on a single image: replicate the image so each copy back-propagates one class
    num_targets = class_indices.numel()
    if image.shape[0] == 1 and num_targets > 1:
        image = image.expand(num_targets, -1, -1, -1)

    # Capture feature maps of the selected stages with forward hooks
    activations = {}
    hooks = []
    for name in layers:
        module = getattr(model, name)
        hooks.append(module.register_forward_hook(
            lambda _m, _inp, out, key=name: activations.__setitem__(key, out)
        ))

    try:
        with torch.enable_grad():
            logits = model(image)                                  # (N, num_classes)
            # Sum of target-class scores; samples are independent (BatchNorm in eval mode)
            score = logits.gather(1, class_indices.view(-1, 1)).sum()
            feats = [activations[name] for name in layers]
            # torch.autograd.grad avoids accumulating gradients into model parameters
            grads = torch.autograd.grad(score, feats)
    finally:
        for h in hooks:
            h.remove()

    fused = None
    for name, A, g in zip(layers, feats, grads):
        # LayerCAM: pixel-wise weights = ReLU(gradient), then M = ReLU(sum_k w_k * A_k)
        cam_l = torch.relu((torch.relu(g) * A).sum(dim=1, keepdim=True)).detach()
        cam_l = F.interpolate(cam_l, size=target_size, mode="bilinear", align_corners=False).squeeze(1)
        cam_l = cam_l / cam_l.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

        # Shallow stages have weak responses: rescale with tanh(gamma * M / max(M))
        if name != "layer4":
            cam_l = torch.tanh(shallow_gamma * cam_l)
            cam_l = cam_l / cam_l.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

        # Fuse stages by element-wise maximum
        fused = cam_l if fused is None else torch.maximum(fused, cam_l)

    return fused


def gen_layercam_resnet(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_indices,
    target_size=None,
    layers: tuple[str, ...] = ("layer4",),
    scales: tuple[float, ...] = (1.0,),
    hflip: bool = False,
    shallow_gamma: float = 2.0,
):
    """
    Generate LayerCAM heatmaps (Jiang et al., TIP 2021) for a ResNet model.
    Same broadcasting modes and TTA options as gen_cam_resnet:
    1. Multi-class on a single image: input (3, H, W) or (1, 3, H, W), class_indices (K,) -> (K, H, W)
    2. Single-class per image on a batch: input (B, 3, H, W), class_indices (B,) -> (B, H, W)

    layers: ResNet stages to use, e.g. ("layer4",) or ("layer3", "layer4") or ("layer2", "layer3", "layer4").
    shallow_gamma: tanh scaling factor applied to shallow stages before fusion.
    """
    model.eval()
    device = next(model.parameters()).device

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    class_indices = torch.as_tensor(class_indices, dtype=torch.long, device=device).view(-1)

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    image = normalize(input_tensor)

    orig_h, orig_w = image.shape[-2:]
    if target_size is None:
        target_size = (orig_h, orig_w)

    accum_cam = None
    for s in scales:
        if s == 1.0:
            scaled_img = image
        else:
            sh = max(32, int(round(orig_h * s)))
            sw = max(32, int(round(orig_w * s)))
            scaled_img = F.interpolate(image, size=(sh, sw), mode="bilinear", align_corners=False)

        cam_s = _layercam_single_pass(model, scaled_img, class_indices, layers, target_size, shallow_gamma)
        accum_cam = cam_s if accum_cam is None else (accum_cam + cam_s)

        if hflip:
            flipped_img = torch.flip(scaled_img, dims=[-1])
            cam_f = _layercam_single_pass(model, flipped_img, class_indices, layers, target_size, shallow_gamma)
            accum_cam = accum_cam + torch.flip(cam_f, dims=[-1])

    cam = accum_cam / accum_cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return cam

# def get_bg_score(cams, alpha)

def ensure_cam_size(cams: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    """Ensure a (K, H, W) or (H, W) CAM array has spatial shape `target_size = (H_t, W_t)`.

    Returns `cams` unchanged when `target_size is None` or spatial dimensions already match.
    """
    if torch.is_tensor(cams):
        cams_t = cams.detach()
        squeeze_2d = cams_t.dim() == 2
        if squeeze_2d:
            cams_t = cams_t.unsqueeze(0)
        if target_size is not None and tuple(cams_t.shape[-2:]) != tuple(target_size):
            added_batch = cams_t.dim() < 4
            if added_batch:
                cams_t = cams_t.unsqueeze(0)
            cams_t = F.interpolate(
                cams_t.float(),
                size=tuple(target_size),
                mode="bilinear",
                align_corners=False,
            )
            if added_batch:
                cams_t = cams_t.squeeze(0)
            cams_t = cams_t.clamp(0.0, 1.0)
        out = cams_t.cpu().numpy()
        return out[0] if squeeze_2d else out

    cams_np = np.asarray(cams, dtype=np.float32)
    squeeze_2d = cams_np.ndim == 2
    if squeeze_2d:
        cams_np = cams_np[None, :, :]
    if target_size is None or cams_np.shape[-2:] == tuple(target_size):
        return cams_np[0] if squeeze_2d else cams_np

    cams_t = torch.from_numpy(cams_np)
    added_batch = cams_t.dim() < 4
    if added_batch:
        cams_t = cams_t.unsqueeze(0)
    resized = F.interpolate(
        cams_t,
        size=tuple(target_size),
        mode="bilinear",
        align_corners=False,
    )
    if added_batch:
        resized = resized.squeeze(0)
    resized = resized.clamp_(0.0, 1.0).numpy()
    return resized[0] if squeeze_2d else resized


def gen_cam_pseudo_mask(
    cams,
    class_indices,
    cam_threshhold: float = 0.25,
    cam_threshold: float | None = None,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    if cam_threshold is not None:
        cam_threshhold = cam_threshold
    cams = ensure_cam_size(cams, target_size)
    if torch.is_tensor(class_indices):
        class_indices = class_indices.detach().cpu().numpy()
    else:
        class_indices = np.asarray(class_indices)
    if cams.ndim == 2:
        cams = np.expand_dims(cams, axis=0)

    voc_class_ids = class_indices.astype(np.int64) + 1  # Add 1 to class indices to account for background class
    max_cam = cams.max(axis=0)
    best_k = cams.argmax(axis=0)

    pseudo_mask = np.where(max_cam >= cam_threshhold, voc_class_ids[best_k], 0)
    return pseudo_mask.astype(np.int64)


def select_active_classes(probs, threshold: float = 0.5) -> np.ndarray:
    """Classes whose probability >= threshold; falls back to the single top class when none pass.

    probs: (num_classes,) tensor or array of sigmoid probabilities for ONE image.
    Returns an int64 array of class indices (never empty).
    """
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy()
    probs = np.asarray(probs)
    active = np.where(probs >= threshold)[0]
    if len(active) == 0:
        active = np.array([int(probs.argmax())], dtype=np.int64)
    return active.astype(np.int64)


def compute_cams(
    model: nn.Module,
    img_tensor: torch.Tensor,
    class_indices,
    method: str = "cam",
    target_size: tuple[int, int] | None = None,
    scales: tuple[float, ...] = (1.0,),
    hflip: bool = False,
    layers: tuple[str, ...] = ("layer4",),
    shallow_gamma: float = 2.0,
) -> np.ndarray:
    """Single entry point for CAM / LayerCAM on one image: returns a NumPy array (K, H, W) in [0, 1].

    - `img_tensor`: (3, H_in, W_in) or (1, 3, H_in, W_in) at classifier input size (e.g. 448x448).
    - `target_size`: optional `(H_orig, W_orig)` output size so CAMs are interpolated directly to the
      native image resolution without intermediate aspect-ratio distortion.
    """
    device = next(model.parameters()).device
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device, non_blocking=True)
    class_idx = torch.as_tensor(class_indices, dtype=torch.long, device=device).view(-1)

    if method == "cam":
        cams = gen_cam_resnet(
            model, img_tensor, class_idx, target_size=target_size, scales=scales, hflip=hflip
        )
    elif method == "layercam":
        cams = gen_layercam_resnet(
            model, img_tensor, class_idx, target_size=target_size,
            layers=layers, scales=scales, hflip=hflip, shallow_gamma=shallow_gamma,
        )
    else:
        raise ValueError(f"method must be 'cam' or 'layercam', got: {method}")
    return cams.detach().cpu().numpy()
