import torch.nn as nn
import torchvision.models
import torch
from torchvision import transforms
import torch.nn.functional as F

def gen_cam_resnet_single_class(model: nn.Module, input_tensor: torch.Tensor , class_indices):
    feature_extractor = nn.Sequential(*list(model.children())[:-2]).eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if(input_tensor).dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    normalize = transforms.Normalize(mean=mean, std=std)

    image = normalize(input_tensor)
    with torch.no_grad():
        feature_maps = feature_extractor.forward(image) # (B, 512, 8, 8)
        class_weights = model.fc.weight[class_indices]  # (B, 512)
        weights = class_weights[:, :, None, None]  # (B, 512, 1, 1)
        # Nhân chập và cộng tổng theo 512 kênh: (B, 1, 8, 8)
        raw_cam = (feature_maps * weights).sum(dim=1, keepdim=True)
        cam = torch.relu(raw_cam)

    # Phóng to CAM từ (8, 8) về bằng kích thước ảnh gốc (244, 244)
    cam = F.interpolate(
        cam, 
        size=image.shape[-2:], 
        mode="bilinear", 
        align_corners=False
    ).squeeze(1)
    
    # Chuẩn hóa giá trị về khoảng [0, 1]
    cam = cam / cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    return cam
