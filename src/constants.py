"""Project-wide constants shared by training, CAM generation, evaluation and logging."""

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

# Official MS COCO 2014/2017 80 object ("thing") classes ordered by official category ID.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Official non-contiguous COCO category IDs (1..90) aligned 1-to-1 with COCO_CLASSES (0..79).
COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)
COCO_CATEGORY_ID_TO_INDEX = dict(zip(COCO_CATEGORY_IDS, range(len(COCO_CLASSES))))

NUM_VOC_CLASSES = len(VOC_CLASSES)            # 20 foreground classes (classifier output size)
NUM_SEG_CLASSES = NUM_VOC_CLASSES + 1         # + background (VOC segmentation label space)
NUM_COCO_CLASSES = len(COCO_CLASSES)          # 80 foreground classes (COCO classifier output size)
NUM_COCO_SEG_CLASSES = NUM_COCO_CLASSES + 1   # + background (COCO segmentation label space)
IGNORE_INDEX = 255                            # Boundary / ignored pixels

SUPPORTED_DATASETS = ("voc2012", "coco")

# ImageNet statistics required by the pretrained ResNet / ConvNeXt backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RESNET_BACKBONES = ("resnet18", "resnet50", "resnet101")
# torchvision ships ConvNeXt V1 only (V2 lives in `timm`). All four use ImageNet-1K weights.
CONVNEXT_BACKBONES = ("convnext_tiny", "convnext_small", "convnext_base", "convnext_large")
SUPPORTED_BACKBONES = RESNET_BACKBONES + CONVNEXT_BACKBONES


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset aliases ('voc', 'voc2012', 'coco', 'coco2014', 'coco2017') to 'voc2012' or 'coco'."""
    key = dataset_name.strip().lower()
    if key in ("voc", "voc2012", "pascal_voc", "pascal_voc2012"):
        return "voc2012"
    if key in ("coco", "coco2014", "coco2017", "mscoco", "ms_coco"):
        return "coco"
    raise ValueError(f"Unsupported dataset: {dataset_name!r}. Expected one of {SUPPORTED_DATASETS}.")


def get_dataset_classes(dataset_name: str) -> tuple[str, ...]:
    """Return the foreground class tuple for the requested dataset ('voc2012' or 'coco')."""
    normalized = normalize_dataset_name(dataset_name)
    if normalized == "voc2012":
        return VOC_CLASSES
    return COCO_CLASSES
