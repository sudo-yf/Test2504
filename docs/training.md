# 训练指南

## 1. 训练人脸检测模型（YOLO Family）

```bash
# YOLO Family 示例（yolov8/yolov11/yolov26/face 变体）
uv run python scripts/train_yolo_face.py \
  --model yolov26n.pt \
  --data configs/datasets/face_detection.yaml \
  --epochs 100 \
  --batch 16 \
  --device 0
```

YOLOv26 兼容示例：

```bash
# 将模型名替换为你使用的 YOLOv26 权重
uv run python scripts/train_yolo_face.py \
  --model yolov26n.pt \
  --data configs/datasets/face_detection.yaml \
  --optimizer auto \
  --device 0
```

该脚本支持任意 Ultralytics 兼容权重。

## 2. 情绪分类微调（Fine-tuning）

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

训练产物输出到 `outputs/train/emotion_finetune`：

- `best.pt`：验证集最优模型
- `last.pt`：最后一个 epoch 的模型
