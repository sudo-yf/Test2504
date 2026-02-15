# Datasets

## Download Emotion Datasets from Hugging Face

```bash
# CLI mode (recommended)
uv run --extra train hf download mrm8488/fer2013 --repo-type dataset --local-dir data/raw/fer2013

# Preset: FER2013
uv run python scripts/download_datasets.py --preset fer2013 --local-dir data/raw

# Custom dataset repo
uv run python scripts/download_datasets.py \
  --repo-id your-org/your-emotion-dataset \
  --repo-type dataset \
  --local-dir data/raw
```

Available presets:

- `fer2013`
- `raf_db`
- `affectnet_subset`

## Prepare Detection Dataset (YOLO)

Expected structure:

```text
data/processed/face_det/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Dataset config: `configs/datasets/face_detection.yaml`

## Prepare Emotion Classification Dataset

Expected structure:

```text
data/processed/emotion_cls/
├── train/
│   ├── happy/
│   ├── sad/
│   └── ...
└── val/
    ├── happy/
    ├── sad/
    └── ...
```
