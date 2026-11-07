from __future__ import annotations

import math
import warnings

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlforge.exceptions import DataValidationError
from mlforge.logging_utils import get_logger

logger = get_logger(__name__)
BINARY_CLASSES = 2


def compute_classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[0] != y_true.shape[0]:
        raise DataValidationError(
            f"proba must be (N, C) matching y_true length; got {proba.shape} vs {y_true.shape}."
        )
    n_classes = proba.shape[1]
    labels = list(range(n_classes))
    y_pred = proba.argmax(axis=1)
    average = "binary" if n_classes == BINARY_CLASSES else "macro"

    proba_norm = proba / proba.sum(axis=1, keepdims=True) 
    roc_auc: float | None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if n_classes == BINARY_CLASSES:
                roc_auc = float(roc_auc_score(y_true, proba_norm[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(y_true, proba_norm, multi_class="ovr", labels=labels)
                )
    except ValueError as exc:
        logger.warning("ROC-AUC undefined for this evaluation set: %s", exc)
        roc_auc = None
    if roc_auc is not None and math.isnan(roc_auc):
        logger.warning("ROC-AUC is undefined for this evaluation set (e.g. one class in y_true).")
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "roc_auc": roc_auc,
        "log_loss": float(log_loss(y_true, proba_norm, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_samples": int(y_true.shape[0]),
    }