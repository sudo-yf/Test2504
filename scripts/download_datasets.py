"""Download emotion datasets/assets from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_DATASETS = {
    "fer2013": "mrm8488/fer2013",
    "raf_db": "andyp192/raf-db",
    "affectnet_subset": "emotion-ai/affectnet-sample",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id. If omitted, use --preset",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(DEFAULT_DATASETS.keys()),
        default="fer2013",
        help="Built-in dataset preset",
    )
    parser.add_argument("--repo-type", type=str, default="dataset", choices=["dataset", "model"])
    parser.add_argument("--local-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--token", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_id = args.repo_id or DEFAULT_DATASETS[args.preset]
    local_dir = args.local_dir / repo_id.split("/")[-1]
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type=args.repo_type,
        local_dir=str(local_dir),
        token=args.token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Downloaded {repo_id} -> {local_dir}")


if __name__ == "__main__":
    main()
