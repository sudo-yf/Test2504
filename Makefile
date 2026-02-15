.PHONY: sync sync-train lint format test check run download-data download-data-hf train-face finetune-emotion compare install-models docker-build docker-run

sync:
	uv sync

sync-train:
	uv sync --extra train --extra models --extra dev

lint:
	uv run --no-project --with ruff ruff check main.py scripts src tests

format:
	uv run --no-project --with ruff ruff format main.py scripts src tests

test:
	PYTHONPATH=src uv run --no-project --with pytest --with pyyaml --with python-dotenv pytest -q tests

check: lint test

run:
	uv run python main.py

download-data:
	uv run python scripts/download_datasets.py --preset fer2013 --local-dir data/raw

download-data-hf:
	uv run --extra train hf download mrm8488/fer2013 --repo-type dataset --local-dir data/raw/fer2013

train-face:
	uv run python scripts/train_yolo_face.py --model yolov26n.pt --data configs/datasets/face_detection.yaml --device 0

finetune-emotion:
	uv run python scripts/finetune_emotion.py --data-root data/processed/emotion_cls --device cuda

compare:
	uv run python scripts/compare_models.py --mode webcam --duration 30

install-models:
	uv run python scripts/install_models.py

docker-build:
	docker compose build

docker-run:
	docker compose run --rm emotisense
