import numpy as np
import torch
import torch.nn.functional as F

def _compute_kernel_weights(
    img: torch.Tensor,
    kernel_size: int,
    dilation: int,
    sxy: float,
    srgb: float = None,
) -> torch.Tensor:
    _, _, H, W = img.shape
    pad = kernel_size // 2
    coords = torch.arange(-pad, pad + 1, dtype=torch.float32, device=img.device) * dilation
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    spatial_w = torch.exp(-(xx**2 + yy**2) / (2 * sxy**2)).reshape(1, 1, -1, 1, 1)

    if srgb is None:
        return spatial_w / spatial_w.sum(dim=2, keepdim=True).clamp_min(1e-8)

    img_patches = F.unfold(
        img,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=pad * dilation,
    ).reshape(1, 3, -1, H, W)
    color_diff_sq = ((img.unsqueeze(2) - img_patches) ** 2).sum(dim=1, keepdim=True)
    bilateral_w = spatial_w * torch.exp(-color_diff_sq / (2 * srgb**2))
    return bilateral_w / bilateral_w.sum(dim=2, keepdim=True).clamp_min(1e-8)


def dense_crf_pytorch(
    image_rgb: np.ndarray,
    probs: np.ndarray,
    n_iters: int = 5,
    sxy_gauss: float = 3.0,
    compat_gauss: float = 3.0,
    sxy_bi: float = 50.0,
    srgb_bi: float = 13.0,
    compat_bi: float = 10.0,
    kernel_size_bi: int = 11,
    dilation_bi: int = 5,
) -> np.ndarray:
    """
    Mean-Field DenseCRF inference in PyTorch with two pairwise branches matching pydensecrf:
    1. Smoothness Gaussian Kernel (local: sxy=3, compat=3)
    2. Appearance Bilateral Kernel (long-range with dilation: sxy=50, srgb=13, compat=10, 51x51 receptive field)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.as_tensor(image_rgb, dtype=torch.float32, device=device)
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    _, _, H, W = img.shape

    P = torch.as_tensor(probs, dtype=torch.float32, device=device).unsqueeze(0)
    C = P.shape[1]
    unary = -torch.log(P.clamp_min(1e-6))

    # 1. Local Smoothness Gaussian branch (5x5, dilation=1)
    w_gauss = _compute_kernel_weights(img, kernel_size=5, dilation=1, sxy=sxy_gauss, srgb=None)

    # 2. Long-range Appearance Bilateral branch (11x11 with dilation=5 -> 51x51 receptive field)
    pad_bi = (kernel_size_bi // 2) * dilation_bi
    w_bilat = _compute_kernel_weights(
        img,
        kernel_size=kernel_size_bi,
        dilation=dilation_bi,
        sxy=sxy_bi,
        srgb=srgb_bi,
    )

    Q = P.clone()
    for _ in range(n_iters):
        q_gauss = F.unfold(Q, kernel_size=5, dilation=1, padding=2).reshape(1, C, -1, H, W)
        msg_gauss = (q_gauss * w_gauss).sum(dim=2)

        q_bilat = F.unfold(
            Q,
            kernel_size=kernel_size_bi,
            dilation=dilation_bi,
            padding=pad_bi,
        ).reshape(1, C, -1, H, W)
        msg_bilat = (q_bilat * w_bilat).sum(dim=2)

        Q = torch.softmax(-unary + compat_gauss * msg_gauss + compat_bi * msg_bilat, dim=1)

    return Q.squeeze(0).cpu().numpy()


def get_dense_crf_mask(
    image_rgb: np.ndarray,
    cams: np.ndarray,
    class_indices,
    alpha: float = 4.0,
    n_iters: int = 5,
    sxy_gauss: float = 3.0,
    compat_gauss: float = 3.0,
    sxy_bi: float = 50.0,
    srgb_bi: float = 13.0,
    compat_bi: float = 10.0,
) -> np.ndarray:
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

    bg_score = (1.0 - cams.max(axis=0, keepdims=True)) ** alpha
    cams = np.concatenate([bg_score, cams], axis=0)
    class_indices = class_indices + 1

    H, W, _ = image_rgb.shape

    cams = np.clip(cams, 1e-6, 1.0)
    cams = cams / np.sum(cams, axis=0, keepdims=True)

    Q = dense_crf_pytorch(
        image_rgb=image_rgb,
        probs=cams,
        n_iters=n_iters,
        sxy_gauss=sxy_gauss,
        compat_gauss=compat_gauss,
        sxy_bi=sxy_bi,
        srgb_bi=srgb_bi,
        compat_bi=compat_bi,
    )

    best_k = np.argmax(Q, axis=0).reshape((H, W))

    voc_lut = np.concatenate([[0], class_indices.astype(np.int64)])
    return voc_lut[best_k]
