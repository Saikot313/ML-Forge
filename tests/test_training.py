import json
import numpy as np
import pytest
import torch

from mlforge.checkpointing import CheckpointContext, load_checkpoint, restore_model, save_checkpoint
from mlforge.config import DiagnosticsConfig, ExperimentConfig, TrainingConfig
from mlforge.data.loaders import make_dataloader
from mlforge.exceptions import CheckpointError, NumericalInstabilityError
from mlforge.models.mlp import MLP
from mlforge.training.early_stopping import EarlyStopping
from mlforge.training.pipeline import run_training
from mlforge.training.trainer import Trainer
from tests.conftest import quick_config_dict

CPU = torch.device("cpu")


def test_early_stopping_triggers_after_patience() -> None:
    stopper = EarlyStopping(patience=2)
    assert [stopper.step(v) for v in (1.0, 0.9, 0.95, 0.96)] == [True, True, False, False]
    assert stopper.should_stop and stopper.best == 0.9


def test_early_stopping_disabled_and_min_delta() -> None:
    disabled = EarlyStopping(patience=0)
    for value in (1.0, 1.1, 1.2, 1.3):
        disabled.step(value)
    assert not disabled.should_stop
    delta = EarlyStopping(patience=1, min_delta=0.1)
    delta.step(1.0)
    assert delta.step(0.95) is False and delta.should_stop


def _separable_loaders(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(256, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    return (
        make_dataloader(x, y, 32, True, seed),
        make_dataloader(x, y, 64, False, seed),
    )


def test_trainer_reduces_loss_and_records_history() -> None:
    torch.manual_seed(0)
    train_loader, val_loader = _separable_loaders()
    config = TrainingConfig(epochs=10, learning_rate=0.05, early_stopping_patience=0)
    trainer = Trainer(MLP(4, [8], 2), config, DiagnosticsConfig(), CPU)
    history = trainer.fit(train_loader, val_loader)
    assert history.epochs_completed == 10 and len(history.train_loss) == 10
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.best_val_loss == pytest.approx(min(history.val_loss))
    assert history.val_loss[history.best_epoch - 1] == pytest.approx(history.best_val_loss)


def test_trainer_stops_early() -> None:
    train_loader, val_loader = _separable_loaders()
    config = TrainingConfig(epochs=50, learning_rate=1e-6, early_stopping_patience=2,
                            early_stopping_min_delta=1.0)
    history = Trainer(MLP(4, [8], 2), config, DiagnosticsConfig(), CPU).fit(train_loader, val_loader)
    assert history.stopped_early and history.epochs_completed < 50


def test_trainer_detects_nan_inputs() -> None:
    x = np.full((64, 4), np.nan, dtype=np.float32)
    y = np.zeros(64, dtype=np.int64)
    loader = make_dataloader(x, y, 16, False, 0)
    trainer = Trainer(MLP(4, [8], 2), TrainingConfig(epochs=1), DiagnosticsConfig(), CPU)
    with pytest.raises(NumericalInstabilityError, match="NaN"):
        trainer.fit(loader, loader)


def test_pipeline_outputs_and_result_sections(trained_run) -> None:
    results = trained_run.results
    for section in ("predictive_quality", "numerical_correctness", "inference_performance",
                    "resource_consumption"):
        assert section in results
    assert (trained_run.run_dir / "results.json").is_file()
    assert (trained_run.run_dir / "training_history.png").is_file()
    with (trained_run.run_dir / "results.json").open(encoding="utf-8") as handle:
        assert json.load(handle)["experiment"]["name"] == "test_run"
    assert results["training_history"]["epochs_completed"] == 4
    assert results["resource_consumption"]["model_memory_mb_estimated"] > 0


def test_checkpoint_roundtrip_matches_in_memory_model(trained_run) -> None:
    assert trained_run.results["numerical_correctness"]["checkpoint_roundtrip_vs_in_memory"]["allclose"]
    checkpoint = load_checkpoint(trained_run.checkpoint)
    assert checkpoint["metadata"]["seed"] == 7 and len(checkpoint["feature_names"]) == 8
    model = restore_model(checkpoint, CPU)
    assert not model.training and model.input_dim == 8


def test_training_is_repeatable_on_same_machine(trained_run, tmp_path) -> None:
    cfg = ExperimentConfig.from_dict(quick_config_dict(str(tmp_path), "repeat_run"))
    repeat = run_training(cfg)
    first, second = trained_run.results["training_history"], repeat["training_history"]
    assert second["val_loss"] == pytest.approx(first["val_loss"], rel=1e-6)
    assert second["train_loss"] == pytest.approx(first["train_loss"], rel=1e-6)


def test_different_seed_changes_training(trained_run, tmp_path) -> None:
    raw = quick_config_dict(str(tmp_path), "other_seed")
    raw["seed"] = 8
    other = run_training(ExperimentConfig.from_dict(raw))
    assert other["training_history"]["train_loss"] != trained_run.results["training_history"]["train_loss"]


def test_load_checkpoint_rejects_corrupt_and_incomplete(tmp_path) -> None:
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        load_checkpoint(corrupt)
    incomplete = tmp_path / "incomplete.pt"
    torch.save({"version": 1}, incomplete)
    with pytest.raises(CheckpointError, match="missing keys"):
        load_checkpoint(incomplete)
    wrong_version = tmp_path / "version.pt"
    torch.save({"version": 99, "model_state_dict": {}, "model_init_config": {},
                "preprocessor": {}, "feature_names": []}, wrong_version)
    with pytest.raises(CheckpointError, match="version"):
        load_checkpoint(wrong_version)


def test_save_checkpoint_is_atomic_and_leaves_no_temp_file(tmp_path) -> None:
    model = MLP(4, [4], 2)
    context = CheckpointContext(config={"a": 1}, preprocessor_state={"eps": 1e-8, "fill_values": [0.0],
                                "mean": [0.0], "scale": [1.0]}, feature_names=["f"], seed=1)
    path = save_checkpoint(tmp_path / "ckpt" / "m.pt", model, None, 1, 0.5, {"x": [1.0]}, context)
    assert path.is_file() and not list(tmp_path.rglob("*.tmp"))
    assert load_checkpoint(path)["best_val_loss"] == 0.5


def test_restore_rejects_mismatched_weights(trained_run) -> None:
    checkpoint = load_checkpoint(trained_run.checkpoint)
    checkpoint["model_init_config"]["hidden_dims"] = [3]
    with pytest.raises(CheckpointError, match="do not match"):
        restore_model(checkpoint, CPU)