#!/usr/bin/env python
"""CLI entry for EmotiSense."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emotisense.app import EmotionDetectionApp
from emotisense.config import Config, get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EmotiSense - Real-time emotion detection")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        cfg = get_config(args.config) if args.config else Config()
        app = EmotionDetectionApp(cfg)
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
