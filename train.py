from __future__ import annotations

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on a custom dataset")
    parser.add_argument("--data", required=True, help="Path to data.yaml in YOLO format")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model weights or config")
    parser.add_argument("--project", default="runs/train", help="Project directory for runs")
    parser.add_argument("--name", default="custom", help="Run name under project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

