from __future__ import annotations

import os
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO


def ensure_dir_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def _draw_annotations(
    base_image: Image.Image,
    boxes_xyxy: np.ndarray,
    class_ids: List[int],
    class_names_map: dict,
) -> Image.Image:
    image = base_image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, xyxy in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
        cls_id = int(class_ids[i]) if i < len(class_ids) else -1
        label = class_names_map.get(cls_id, str(cls_id))

        # Box
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)

        # Label background
        text = label
        if font is not None:
            text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
        else:
            text_w, text_h = draw.textlength(text), 14
        pad = 2
        draw.rectangle(
            [(x1, max(0, y1 - text_h - 2 * pad)), (x1 + text_w + 2 * pad, y1)],
            fill=(0, 128, 0),
        )
        # Text
        draw.text((x1 + pad, max(0, y1 - text_h - pad)), text, fill=(255, 255, 255), font=font)

    return image


def run_yolo_detection(
    image_path: str,
    output_dir: str,
    model_weights_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
) -> Tuple[str, List[str]]:
    """
    Run object detection on the provided image using a YOLO model and save an annotated result image.

    Returns a tuple of (output_image_path, detected_class_names)
    """
    ensure_dir_exists(output_dir)

    model = YOLO(model_weights_path)
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        verbose=False,
        save=False,
    )

    if not results:
        raise RuntimeError("YOLO returned no results.")

    result = results[0]

    # Extract detected class names and boxes
    class_names: List[str] = []
    boxes_xyxy: np.ndarray = np.empty((0, 4))
    class_ids: List[int] = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.int().tolist()
        names_map = result.names or {}
        class_names = [names_map.get(cls_id, str(cls_id)) for cls_id in class_ids]
    else:
        names_map = result.names or {}

    # Open base image and draw annotations (no OpenCV)
    base_image = Image.open(image_path).convert("RGB")
    annotated_image = _draw_annotations(base_image, boxes_xyxy, class_ids, names_map)

    # Build output path
    input_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_detected{ext or '.jpg'}"
    output_path = os.path.join(output_dir, output_filename)

    # Save annotated image
    annotated_image.save(output_path, quality=95)

    return output_path, class_names

