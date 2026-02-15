"""Unified CLI for EmotiSense product workflows."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EmotiSense 产品 CLI")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run", help="运行实时检测")

    doctor_parser = subparsers.add_parser("doctor", help="运行环境自检")
    doctor_parser.add_argument("--check-camera", action="store_true", help="检查摄像头可用性")

    report_parser = subparsers.add_parser("report", help="从现有日志/数据库生成报告")
    report_parser.add_argument("--mode", type=str, default="offline", choices=["offline", "realtime"])
    return parser


def _load_config(config_path: str | None) -> Config:
    return Config(config_path) if config_path else Config()


def _print_doctor_result(result: dict[str, Any]) -> None:
    print("\nEmotiSense Doctor")
    print("=" * 60)
    for item in result["checks"]:
        print(f"[{item['status'].upper():5s}] {item['name']:18s} {item['detail']}")
    print("-" * 60)
    print(
        "Summary: "
        f"pass={result['passed']} warn={result['warned']} fail={result['failed']} skip={result['skipped']}"
    )


def run_cli(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"
    config = _load_config(args.config)

    if command == "run":
        from .app import EmotionDetectionApp

        app = EmotionDetectionApp(config)
        app.run()
        return 0

    if command == "doctor":
        from .product_ops import ProductDoctor

        doctor = ProductDoctor(config)
        result = doctor.run(check_camera=bool(args.check_camera))
        _print_doctor_result(result)
        return 1 if result["failed"] > 0 else 0

    if command == "report":
        from .data_manager import EmotionDataManager
        from .product_ops import SessionReporter

        manager = EmotionDataManager(config)
        reporter = SessionReporter(config)
        exported = reporter.export(manager, mode=args.mode)
        print(f"JSON 报告: {exported['json_path']}")
        print(f"Markdown 报告: {exported['md_path']}")
        return 0

    logger.error("Unknown command: %s", command)
    return 1


def main() -> None:
    try:
        exit_code = run_cli()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit_code = 130
    raise SystemExit(exit_code)
