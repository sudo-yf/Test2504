#!/usr/bin/env python
"""CLI entry for EmotiSense."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from emotisense.cli import main as cli_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        cli_main()
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
