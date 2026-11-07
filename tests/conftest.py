from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlforge.config import ExperimentConfig, InferenceConfig
from mlforge.inference.predictor import Predictor
from mlforge.training.pipeline import run_training


def quick_config_dict(output_dir: str, experiment_name: str = "test_run") -> dict[str, Any]:
    return {
        "seed": 7,
        "deterministic": True,
        "device": "cpu",
        "output_dir": output_dir,
        "experiment_name": experiment_name,
        "data": {
            "n_samples": 400,
            "n_features": 8,
            "n_informative": 5,
            "n_redundant": 1,
            "n_classes": 2,
            "missing_fraction": 0.02,
        },
        "model": {"hidden_dims": [8, 4], "activation": "relu", "dropout": 0.0},
        "training": {
            "epochs": 4,
            "batch_size": 64,
            "learning_rate": 0.01,
            "early_stopping_patience": 0,
        },
        "inference": {"device": "cpu", "batch_size": 32},
        "benchmark": {
            "warmup_iterations": 1,
            "iterations": 2,
            "batch_sizes": [1, 8],
            "num_samples": 32,
            "devices": ["cpu"],
        },
    }


@pytest.fixture
def quick_cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(quick_config_dict(str(tmp_path)))


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    out_dir = tmp_path_factory.mktemp("mlforge_run")
    cfg = ExperimentConfig.from_dict(quick_config_dict(str(out_dir)))
    results = run_training(cfg)
    return SimpleNamespace(
        cfg=cfg,
        results=results,
        checkpoint=Path(results["artifacts"]["checkpoint"]),
        run_dir=out_dir / cfg.experiment_name,
    )


@pytest.fixture
def predictor(trained_run: SimpleNamespace) -> Predictor:
    return Predictor.from_checkpoint(
        trained_run.checkpoint, InferenceConfig(device="cpu", batch_size=32)
    )