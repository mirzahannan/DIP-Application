from __future__ import annotations

import os
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

# COCO 91 classes with background index 0 -> labels start at 1 in torchvision
COCO_CLASSES: List[str] = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def ensure_dir_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def _draw_annotations(image: Image.Image, boxes, labels, scores, conf: float) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box, label_idx, score in zip(boxes, labels, scores):
        if float(score) < conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        class_name = COCO_CLASSES[int(label_idx)] if int(label_idx) < len(COCO_CLASSES) else str(label_idx)
        text = f"{class_name} {float(score):.2f}"

        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
        if font is not None:
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        else:
            tw, th = draw.textlength(text), 14
        pad = 2
        draw.rectangle([(x1, max(0, y1 - th - 2 * pad)), (x1 + tw + 2 * pad, y1)], fill=(0, 128, 0))
        draw.text((x1 + pad, max(0, y1 - th - pad)), text, fill=(255, 255, 255), font=font)

    return img


def run_torchvision_detection(
    image_path: str,
    output_dir: str,
    confidence_threshold: float = 0.25,
) -> Tuple[str, List[str]]:
    ensure_dir_exists(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = to_tensor(image).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    annotated = _draw_annotations(image, boxes, labels, scores, confidence_threshold)

    # Detected label names above threshold
    detected_names: List[str] = []
    for label_idx, score in zip(labels.tolist(), scores.tolist()):
        if float(score) >= confidence_threshold:
            name = COCO_CLASSES[int(label_idx)] if int(label_idx) < len(COCO_CLASSES) else str(label_idx)
            detected_names.append(name)

    name, ext = os.path.splitext(os.path.basename(image_path))
    out_path = os.path.join(output_dir, f"{name}_detected{ext or '.jpg'}")
    annotated.save(out_path, quality=95)

    return out_path, detected_names
