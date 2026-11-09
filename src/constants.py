"""Project-wide constants shared by training, CAM generation, evaluation and logging."""

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

NUM_VOC_CLASSES = len(VOC_CLASSES)      # 20 foreground classes (classifier output size)
NUM_SEG_CLASSES = NUM_VOC_CLASSES + 1   # + background (segmentation label space)
IGNORE_INDEX = 255                      # VOC boundary / "difficult" pixels

# ImageNet statistics required by the pretrained ResNet backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "resnet101")
