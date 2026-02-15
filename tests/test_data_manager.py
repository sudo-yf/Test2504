from pathlib import Path

from emotisense.config import Config
from emotisense.data_manager import EmotionDataManager


def test_data_manager_persists_to_sqlite(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
video: {}
face_detection: {}
eye_detection: {}
emotion:
  max_data_records: 100

data:
  log_file: "emotion_log.txt"
  database_path: "data/test.db"
  cleanup_interval: 1
  enable_logging: false

api: {}
ui: {}
""".strip(),
        encoding="utf-8",
    )

    cfg = Config(str(cfg_file))
    manager = EmotionDataManager(cfg)
    manager.add_record("happy", 98.0)

    stats = manager.get_db_statistics()
    assert stats["total_records"] >= 1
    assert "happy" in stats["emotions"]
