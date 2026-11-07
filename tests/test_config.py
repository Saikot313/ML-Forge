from pathlib import Path

import pytest

from mlforge.config import ExperimentConfig, load_config
from mlforge.exceptions import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def test_default_and_quick_yaml_files_load() -> None:
    default = load_config(ROOT / "configs" / "default.yaml")
    quick = load_config(ROOT / "configs" / "quick_test.yaml")
    assert default.model.hidden_dims == (64, 32)
    assert quick.seed == 7 and quick.device == "cpu"


def test_none_path_returns_defaults() -> None:
    assert load_config(None) == ExperimentConfig()


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_unknown_top_level_key_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown top-level"):
        ExperimentConfig.from_dict({"sed": 1})


def test_unknown_section_key_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown keys"):
        ExperimentConfig.from_dict({"training": {"epoch": 3}})


def test_invalid_values_rejected() -> None:
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"model": {"activation": "swishy"}})
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"data": {"val_size": 0.5, "test_size": 0.5}})
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"training": {"learning_rate": 0}})
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"benchmark": {"batch_sizes": [8], "num_samples": 4}})


def test_dict_roundtrip() -> None:
    cfg = ExperimentConfig()
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(bad)