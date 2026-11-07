import numpy as np
import pytest

from mlforge.evaluation import compare_models, compute_classification_metrics, prediction_agreement
from mlforge.exceptions import DataValidationError


def test_binary_metrics_hand_computed() -> None:
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    m = compute_classification_metrics(y_true, proba)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(0.8)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["log_loss"] > 0 and m["n_samples"] == 4


def test_multiclass_metrics() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    proba = np.full((6, 3), 0.1)
    proba[np.arange(6), y_true] = 0.8
    m = compute_classification_metrics(y_true, proba)
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0 and m["roc_auc"] == pytest.approx(1.0)
    assert np.trace(np.array(m["confusion_matrix"])) == 6


def test_roc_auc_none_when_undefined() -> None:
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    m = compute_classification_metrics(np.zeros(3, dtype=int), proba)
    assert m["roc_auc"] is None and m["accuracy"] == 1.0


def test_metrics_validate_shapes() -> None:
    with pytest.raises(DataValidationError):
        compute_classification_metrics(np.array([0, 1]), np.array([[0.5, 0.5]]))


def test_compare_models_is_descriptive() -> None:
    metrics = {
        "a": {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1": 0.8, "roc_auc": None, "log_loss": 0.5},
        "b": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9, "roc_auc": 0.95, "log_loss": 0.3},
    }
    report = compare_models(metrics, "a", "b")
    assert report["differences"]["accuracy"]["candidate_minus_reference"] == pytest.approx(0.1)
    assert report["differences"]["log_loss"]["direction"] == "lower_is_better"
    assert report["differences"]["roc_auc"] is None
    assert "winner" not in report
    with pytest.raises(DataValidationError):
        compare_models(metrics, "a", "missing")


def test_prediction_agreement() -> None:
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    same = prediction_agreement(proba, proba)
    assert same["label_agreement"] == 1.0 and same["max_abs_prob_diff"] == 0.0
    flipped = prediction_agreement(proba, proba[:, ::-1])
    assert flipped["label_agreement"] == 0.0
    with pytest.raises(DataValidationError):
        prediction_agreement(proba, np.zeros((3, 2)))


def test_pipeline_reports_both_models(trained_run) -> None:
    metrics = trained_run.results["predictive_quality"]["test_metrics"]
    assert set(metrics) == {"mlp", "logistic_regression"}
    for values in metrics.values():
        assert 0.0 <= values["accuracy"] <= 1.0
        assert np.array(values["confusion_matrix"]).sum() == values["n_samples"]