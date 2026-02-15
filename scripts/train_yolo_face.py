"""Train YOLO-family face detector (YOLOv8/11/26-compatible checkpoints)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

"""
Notes:
- Any Ultralytics-compatible checkpoint can be passed via --model.
- For YOLOv26, use corresponding checkpoint name when available in your environment.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face detector with YOLO family")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--data", type=Path, default=Path("configs/datasets/face_detection.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--project", type=Path, default=Path("outputs/train/face_detector"))
    parser.add_argument("--name", type=str, default="yolo_face")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        project=str(args.project),
        name=args.name,
    )


if __name__ == "__main__":
    main()
