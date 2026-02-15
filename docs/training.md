# Training

## 1. Train Face Detector (YOLO-family)

```bash
# YOLO family example (yolov8/yolov11/yolov26/face variants)
uv run python scripts/train_yolo_face.py \
  --model yolov26n.pt \
  --data configs/datasets/face_detection.yaml \
  --epochs 100 \
  --batch 16 \
  --device 0
```

YOLOv26 compatibility:

```bash
# Replace with your YOLOv26-compatible checkpoint name
uv run python scripts/train_yolo_face.py \
  --model yolov26n.pt \
  --data configs/datasets/face_detection.yaml \
  --optimizer auto \
  --device 0
```

The script supports any Ultralytics-compatible model checkpoint.

## 2. Fine-tune Emotion Classifier

```bash
uv run python scripts/finetune_emotion.py \
  --data-root data/processed/emotion_cls \
  --model resnet18 \
  --num-classes 7 \
  --epochs 20 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda
```

Outputs are saved to `outputs/train/emotion_finetune`:

- `best.pt` (best validation accuracy)
- `last.pt` (final epoch checkpoint)

## 3. Model Comparison

```bash
uv run python scripts/compare_models.py --mode webcam --duration 30
```
