# EmotiSense

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%20~%203.11-blue.svg)](https://www.python.org/)
[![UV](https://img.shields.io/badge/env-uv-6e56cf.svg)](https://docs.astral.sh/uv/)
[![PyTorch](https://img.shields.io/badge/training-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/detection-YOLO%20Family-111111.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

面向实时情绪识别与训练实验的一体化项目，提供推理、数据管理、可视化分析、YOLO 系列人脸检测训练与情绪分类微调流程。项目使用统一工程化结构，支持本地开发与 Docker 部署。

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Quick Start](#quick-start)
  - [1. Installation](#1-installation)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Train](#3-train)
  - [4. Run Inference](#4-run-inference)
  - [5. Quality Gates](#5-quality-gates)
- [GPU Requirements](#gpu-requirements)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Versioning](#versioning)

## Quick Start

### 1. Installation

#### Use UV (Recommended)

```bash
uv sync
cp .env.example .env
```

安装训练与扩展模型依赖：

```bash
uv sync --extra train --extra models --extra dev
```

#### Use Docker

```bash
docker compose build
docker compose run --rm emotisense
```

### 2. Data Preparation

#### Download Emotion Dataset (HF CLI)

```bash
uv run --extra train hf download mrm8488/fer2013 \
  --repo-type dataset \
  --local-dir data/raw/fer2013
```

#### Download Emotion Dataset (Script)

```bash
uv run python scripts/download_datasets.py --preset fer2013 --local-dir data/raw
```

支持自定义数据集仓库：

```bash
uv run python scripts/download_datasets.py \
  --repo-id your-org/your-emotion-dataset \
  --repo-type dataset \
  --local-dir data/raw
```

### 3. Train

#### Train Face Detector (YOLO Family, including YOLOv26 variants)

```bash
uv run python scripts/train_yolo_face.py \
  --model yolov26n.pt \
  --data configs/datasets/face_detection.yaml \
  --epochs 100 \
  --device 0
```

说明：`--model` 支持任意 Ultralytics 兼容权重（如 `yolov8*`、`yolov11*`、`yolov26*` 或 face 变体权重）。

#### Fine-tune Emotion Classifier

```bash
uv run python scripts/finetune_emotion.py \
  --data-root data/processed/emotion_cls \
  --model resnet18 \
  --num-classes 7 \
  --epochs 20 \
  --device cuda
```

输出目录：`outputs/train/emotion_finetune`

### 4. Run Inference

```bash
uv run python main.py
```

模型对比（摄像头）：

```bash
uv run python scripts/compare_models.py --mode webcam --duration 30
```

### 5. Quality Gates

```bash
make lint
make test
make check
```

## GPU Requirements

- 训练推荐 NVIDIA GPU + CUDA 环境
- 人脸检测训练（YOLO）：8GB+ 显存
- 情绪分类微调：8GB+ 显存
- 多模型并行实验或更大 batch：12GB+ 显存
- CPU 可用于基础推理演示，但吞吐明显低于 GPU

## Project Structure

```text
Test2504/
├── src/emotisense/            # Core app modules
├── scripts/                   # Data download / train / finetune / comparison
├── configs/                   # Training and dataset configs
├── docs/                      # Detailed guides
├── tests/                     # Unit tests
├── data/                      # Local data root (raw/processed)
├── main.py                    # Runtime entry
├── config.yaml                # Runtime configuration
├── pyproject.toml             # Dependency and tooling configuration
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## Documentation

- [Installation](docs/installation.md)
- [Datasets](docs/datasets.md)
- [Training](docs/training.md)
- [GPU Requirements](docs/gpu_requirements.md)
- [Policy / Evaluation](docs/policy_eval.md)
- [Advanced Models](docs/ADVANCED_MODELS.md)
- [Model Internals](docs/MODELS_IMPLEMENTATION.md)

## Versioning

发布与版本策略见 [RELEASE.md](RELEASE.md)。
