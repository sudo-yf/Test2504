"""Product-grade operational utilities: doctor checks and report export."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Config
from .data_manager import EmotionDataManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .analyzer import EmotionAnalyzer


class _NoopAnalyzer:
    """Fallback analyzer used when API dependencies are unavailable."""

    @staticmethod
    def analyze_emotion_logs(log_lines: list[str]) -> None:
        return None


@dataclass
class CheckResult:
    name: str
    status: str  # pass / warn / fail / skip
    detail: str


class ProductDoctor:
    """Run environment and runtime readiness checks."""

    def __init__(self, config: Config):
        self.config = config

    def run(self, check_camera: bool = False) -> dict[str, Any]:
        checks: list[CheckResult] = []
        checks.append(self._check_config_file())
        checks.append(self._check_database_path())
        checks.append(self._check_log_path())
        checks.append(self._check_api_key())
        checks.append(self._check_camera(check_camera))

        summary = {
            "total": len(checks),
            "passed": sum(1 for c in checks if c.status == "pass"),
            "warned": sum(1 for c in checks if c.status == "warn"),
            "failed": sum(1 for c in checks if c.status == "fail"),
            "skipped": sum(1 for c in checks if c.status == "skip"),
            "checks": [asdict(c) for c in checks],
        }
        return summary

    def _check_config_file(self) -> CheckResult:
        if self.config.config_path.exists():
            return CheckResult("config_file", "pass", f"found: {self.config.config_path}")
        return CheckResult("config_file", "fail", f"missing: {self.config.config_path}")

    def _check_database_path(self) -> CheckResult:
        db_path = Path(self.config.get("data.database_path", "data/emotions.db"))
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db_path) as conn:
                conn.execute("SELECT 1")
            return CheckResult("database_path", "pass", f"ready: {db_path}")
        except Exception as exc:
            return CheckResult("database_path", "fail", f"{db_path}: {exc}")

    def _check_log_path(self) -> CheckResult:
        log_path = Path(self.config.get("data.log_file", "emotion_log.txt"))
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8"):
                pass
            return CheckResult("log_path", "pass", f"writable: {log_path}")
        except Exception as exc:
            return CheckResult("log_path", "fail", f"{log_path}: {exc}")

    def _check_api_key(self) -> CheckResult:
        if self.config.deepseek_api_key:
            return CheckResult("deepseek_api_key", "pass", "configured")
        return CheckResult("deepseek_api_key", "warn", "missing; trend analysis API will be skipped")

    def _check_camera(self, check_camera: bool) -> CheckResult:
        if not check_camera:
            return CheckResult("camera", "skip", "not requested")

        try:
            import cv2  # Lazy import to keep doctor/report usable without OpenCV installed.
        except ImportError:
            return CheckResult("camera", "fail", "opencv-python is not installed")

        camera_index = int(self.config.get("video.camera_index", 0))
        cap = cv2.VideoCapture(camera_index)
        try:
            if not cap.isOpened():
                return CheckResult("camera", "fail", f"cannot open camera index {camera_index}")
            return CheckResult("camera", "pass", f"camera index {camera_index} is available")
        finally:
            cap.release()


class SessionReporter:
    """Generate machine-readable and human-readable session reports."""

    def __init__(self, config: Config, analyzer: "EmotionAnalyzer | None" = None):
        self.config = config
        self.analyzer = analyzer or self._build_default_analyzer()
        self.reports_dir = Path(config.get("product.reports_dir", "outputs/reports"))
        self.sample_size = int(config.get("product.report_sample_size", 500))

    def _build_default_analyzer(self):
        try:
            from .analyzer import EmotionAnalyzer

            return EmotionAnalyzer(self.config)
        except Exception as exc:
            logger.warning("Analyzer unavailable, fallback to no-op analyzer: %s", exc)
            return _NoopAnalyzer()

    def export(self, manager: EmotionDataManager, mode: str = "realtime") -> dict[str, Any]:
        snapshot = self._build_snapshot(manager, mode=mode)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        json_path = self.reports_dir / f"{mode}-report-{timestamp}.json"
        md_path = self.reports_dir / f"{mode}-report-{timestamp}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._to_markdown(snapshot))

        return {"json_path": str(json_path), "md_path": str(md_path), "snapshot": snapshot}

    def _build_snapshot(self, manager: EmotionDataManager, mode: str) -> dict[str, Any]:
        memory_stats = manager.get_statistics()
        db_stats = manager.get_db_statistics()
        log_lines = manager.read_log_file()
        sampled_logs = log_lines[-self.sample_size :]

        analysis = None
        if sampled_logs:
            analysis = self.analyzer.analyze_emotion_logs(sampled_logs)

        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "mode": mode,
            "config_path": str(self.config.config_path),
            "memory_stats": memory_stats,
            "db_stats": db_stats,
            "log_lines": len(log_lines),
            "analysis_text": analysis,
        }

    @staticmethod
    def _to_markdown(snapshot: dict[str, Any]) -> str:
        db_stats = snapshot.get("db_stats", {})
        memory_stats = snapshot.get("memory_stats", {})
        analysis_text = snapshot.get("analysis_text") or "未生成（缺少 API Key 或日志为空）"

        return "\n".join(
            [
                "# EmotiSense 分析报告",
                "",
                f"- 生成时间: {snapshot.get('generated_at', '-')}",
                f"- 运行模式: {snapshot.get('mode', '-')}",
                f"- 配置文件: `{snapshot.get('config_path', '-')}`",
                "",
                "## 数据概览",
                f"- 内存记录数: {memory_stats.get('total_records', 0)}",
                f"- 数据库记录数: {db_stats.get('total_records', 0)}",
                f"- 数据库平均置信度: {db_stats.get('average_confidence', 0.0):.2f}",
                f"- 日志行数: {snapshot.get('log_lines', 0)}",
                "",
                "## 情绪分布（数据库）",
                json.dumps(db_stats.get("emotions", {}), ensure_ascii=False),
                "",
                "## 趋势分析",
                analysis_text,
                "",
            ]
        )
