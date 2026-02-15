import json
from pathlib import Path

from emotisense.config import Config
from emotisense.data_manager import EmotionDataManager
from emotisense.product_ops import ProductDoctor, SessionReporter


def _write_config(path: Path) -> None:
    path.write_text(
        """
video:
  camera_index: 0
face_detection: {}
eye_detection: {}
emotion:
  max_data_records: 100
data:
  log_file: "data/test.log"
  database_path: "data/test.db"
  cleanup_interval: 1
  enable_logging: true
api:
  enabled: false
ui: {}
product:
  reports_dir: "outputs/reports"
  report_sample_size: 10
""".strip(),
        encoding="utf-8",
    )


def test_product_doctor_basic_checks(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    _write_config(cfg_file)

    cfg = Config(str(cfg_file))
    doctor = ProductDoctor(cfg)
    result = doctor.run(check_camera=False)

    assert result["failed"] == 0
    names = {item["name"] for item in result["checks"]}
    assert "config_file" in names
    assert "database_path" in names
    assert "log_path" in names


def test_session_reporter_exports_json_and_markdown(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    _write_config(cfg_file)

    cfg = Config(str(cfg_file))
    manager = EmotionDataManager(cfg)
    manager.add_record("happy", 98.0)
    manager.append_to_log("2026-01-01 00:00:00 检测到高强度情绪：happy (98%)")

    reporter = SessionReporter(cfg)
    exported = reporter.export(manager, mode="offline")

    json_path = Path(exported["json_path"])
    md_path = Path(exported["md_path"])
    assert json_path.exists()
    assert md_path.exists()

    snapshot = json.loads(json_path.read_text(encoding="utf-8"))
    assert snapshot["db_stats"]["total_records"] >= 1
    assert "happy" in snapshot["db_stats"]["emotions"]

