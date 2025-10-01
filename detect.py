from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def ensure_dir_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


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

    # Extract detected class names
    class_names: List[str] = []
    if result.boxes is not None and len(result.boxes) > 0:
        class_ids = result.boxes.cls.int().tolist()
        names_map = result.names or {}
        class_names = [names_map.get(cls_id, str(cls_id)) for cls_id in class_ids]

    # Create annotated image using Ultralytics' built-in plotting
    annotated_bgr: np.ndarray = result.plot()

    # Build output path
    input_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_detected{ext or '.jpg'}"
    output_path = os.path.join(output_dir, output_filename)

    # Save annotated image
    cv2.imwrite(output_path, annotated_bgr)

    return output_path, class_names

