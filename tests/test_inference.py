import numpy as np
import pytest

from mlforge.config import InferenceConfig
from mlforge.exceptions import CheckpointError, DataValidationError, NumericalInstabilityError
from mlforge.inference import Predictor
from mlforge.numerics.stability import compare_arrays


def raw_batch(predictor: Predictor, n: int = 10) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, predictor.n_features))


def test_probabilities_are_valid(predictor: Predictor) -> None:
    proba = predictor.predict_proba(raw_batch(predictor))
    assert proba.shape == (10, predictor.n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert (proba >= 0).all()


def test_single_sample_matches_batch_row(predictor: Predictor) -> None:
    batch = raw_batch(predictor)
    single = predictor.predict_one(list(batch[3]))
    np.testing.assert_allclose(single["probabilities"], predictor.predict_proba(batch)[3], atol=1e-6)
    assert single["label"] == int(np.argmax(single["probabilities"]))


def test_predictions_invariant_to_batch_size(predictor: Predictor) -> None:
    features = predictor.preprocessor.transform(raw_batch(predictor, 50))
    small = predictor.predict_proba_preprocessed(features, batch_size=7)
    large = predictor.predict_proba_preprocessed(features, batch_size=1000)
    assert compare_arrays(small, large, rtol=1e-5, atol=1e-6).allclose


def test_raw_and_preprocessed_paths_agree(predictor: Predictor) -> None:
    raw = raw_batch(predictor)
    via_raw = predictor.predict_proba(raw)
    via_pre = predictor.predict_proba_preprocessed(predictor.preprocessor.transform(raw))
    np.testing.assert_array_equal(via_raw, via_pre)


def test_missing_values_are_imputed(predictor: Predictor) -> None:
    row = [0.0] * predictor.n_features
    row[2] = None
    assert np.isfinite(predictor.predict_one(row)["probabilities"]).all()


def test_input_validation(predictor: Predictor) -> None:
    with pytest.raises(DataValidationError, match="expected"):
        predictor.predict_proba([[0.0] * (predictor.n_features + 1)])
    with pytest.raises(DataValidationError, match="2-D"):
        predictor.predict_proba([0.0] * predictor.n_features)
    with pytest.raises(DataValidationError, match="infinite"):
        predictor.predict_proba([[np.inf] + [0.0] * (predictor.n_features - 1)])
    with pytest.raises(DataValidationError, match="no rows"):
        predictor.predict_proba(np.empty((0, predictor.n_features)))


def test_empty_preprocessed_input(predictor: Predictor) -> None:
    out = predictor.predict_logits_preprocessed(np.empty((0, predictor.n_features), dtype=np.float32))
    assert out.shape == (0, predictor.n_classes)


def test_inference_mode_and_baseline_agree(trained_run) -> None:
    fast = Predictor.from_checkpoint(trained_run.checkpoint, InferenceConfig(device="cpu"))
    slow = Predictor.from_checkpoint(
        trained_run.checkpoint, InferenceConfig(device="cpu", use_inference_mode=False)
    )
    raw = raw_batch(fast, 20)
    assert compare_arrays(fast.predict_proba(raw), slow.predict_proba(raw), rtol=1e-6, atol=1e-8).allclose


def test_amp_disabled_on_cpu(trained_run) -> None:
    amp = Predictor.from_checkpoint(trained_run.checkpoint, InferenceConfig(device="cpu", use_amp=True))
    assert amp.use_amp is False


def test_adaptive_batch_size_halves_on_oom(predictor: Predictor, monkeypatch) -> None:
    original = predictor._forward_all
    sizes: list[int] = []

    def flaky(features, batch_size):
        sizes.append(batch_size)
        if batch_size > 4:
            raise RuntimeError("CUDA out of memory (simulated)")
        return original(features, batch_size)

    monkeypatch.setattr(predictor, "_forward_all", flaky)
    logits = predictor.predict_logits_preprocessed(
        np.zeros((10, predictor.n_features), dtype=np.float32), batch_size=16
    )
    assert sizes == [16, 8, 4] and logits.shape == (10, predictor.n_classes)


def test_non_oom_errors_propagate(predictor: Predictor, monkeypatch) -> None:
    def broken(features, batch_size):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(predictor, "_forward_all", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        predictor.predict_logits_preprocessed(np.zeros((2, predictor.n_features), dtype=np.float32))


def test_oom_not_retried_when_adaptive_disabled(trained_run, monkeypatch) -> None:
    strict = Predictor.from_checkpoint(
        trained_run.checkpoint, InferenceConfig(device="cpu", adaptive_batch_size=False)
    )

    def oom(features, batch_size):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(strict, "_forward_all", oom)
    with pytest.raises(RuntimeError):
        strict.predict_logits_preprocessed(np.zeros((2, strict.n_features), dtype=np.float32))


def test_invalid_model_output_is_rejected(predictor: Predictor, monkeypatch) -> None:
    monkeypatch.setattr(
        predictor, "_forward_all",
        lambda features, batch_size: np.full((features.shape[0], predictor.n_classes), np.nan, np.float32),
    )
    with pytest.raises(NumericalInstabilityError):
        predictor.predict_logits_preprocessed(np.zeros((2, predictor.n_features), dtype=np.float32))


def test_missing_checkpoint(tmp_path) -> None:
    with pytest.raises(CheckpointError):
        Predictor.from_checkpoint(tmp_path / "none.pt")