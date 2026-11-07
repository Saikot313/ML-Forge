from __future__ import annotations

from typing import Any

import numpy as np

from mlforge.exceptions import DataValidationError

COMPARED_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss")
LOWER_IS_BETTER = {"log_loss"}


def compare_models(
    metrics_by_model: dict[str, dict[str, Any]],
    reference: str,
    candidate: str,
) -> dict[str, Any]:
    for name in (reference, candidate):
        if name not in metrics_by_model:
            raise DataValidationError(f"Model '{name}' missing from metrics.")
    table: dict[str, dict[str, Any]] = {}
    differences: dict[str, Any] = {}
    for metric in COMPARED_METRICS:
        ref_value = metrics_by_model[reference].get(metric)
        cand_value = metrics_by_model[candidate].get(metric)
        table[metric] = {reference: ref_value, candidate: cand_value}
        if ref_value is None or cand_value is None:
            differences[metric] = None
        else:
            differences[metric] = {
                "candidate_minus_reference": float(cand_value - ref_value),
                "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
            }
    return {
        "reference": reference,
        "candidate": candidate,
        "metrics": table,
        "differences": differences,
        "note": "Single-split, single-seed comparison; differences may be within run-to-run variance.",
    }


def prediction_agreement(proba_a: np.ndarray, proba_b: np.ndarray) -> dict[str, float]:
    a = np.asarray(proba_a, dtype=np.float64)
    b = np.asarray(proba_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise DataValidationError(f"Probability shapes must match and be 2-D; got {a.shape}, {b.shape}.")
    diff = np.abs(a - b)
    return {
        "label_agreement": float((a.argmax(axis=1) == b.argmax(axis=1)).mean()),
        "max_abs_prob_diff": float(diff.max()),
        "mean_abs_prob_diff": float(diff.mean()),
    }