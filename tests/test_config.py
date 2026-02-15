from pathlib import Path

from emotisense.config import Config


def test_load_custom_config(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("video:\n  frame_width: 320\n", encoding="utf-8")

    cfg = Config(str(cfg_file))
    assert cfg.get("video.frame_width") == 320


def test_missing_key_returns_default(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("video: {}\n", encoding="utf-8")
    cfg = Config(str(cfg_file))
    assert cfg.get("not.exists", "fallback") == "fallback"
