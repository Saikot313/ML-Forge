from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mlforge.exceptions import DataValidationError

MIN_PER_CLASS = 3


@dataclass(frozen=True)
class DataReport:
    n_rows: int
    n_features: int
    n_classes: int
    class_counts: dict[int, int]
    missing_fraction: float
    columns_with_missing: list[str]
    feature_names: list[str]


def validate_dataframe(
    frame: pd.DataFrame,
    target_column: str,
    expected_n_features: int | None = None,
    max_missing_fraction: float = 0.5,
    min_rows: int = 50,
) -> DataReport:
    if frame.empty:
        raise DataValidationError("Dataset is empty.")
    if target_column not in frame.columns:
        raise DataValidationError(f"Target column '{target_column}' not found.")
    if len(frame) < min_rows:
        raise DataValidationError(f"Dataset has {len(frame)} rows; at least {min_rows} required.")

    target = frame[target_column]
    if not pd.api.types.is_numeric_dtype(target):
        raise DataValidationError("Target column must be integer-coded (0..K-1).")
    if target.isna().any():
        raise DataValidationError("Target column contains missing values.")

    features = frame.drop(columns=[target_column])
    if features.shape[1] == 0:
        raise DataValidationError("Dataset has no feature columns.")
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric feature columns: {non_numeric}")
    if expected_n_features is not None and features.shape[1] != expected_n_features:
        raise DataValidationError(
            f"Expected {expected_n_features} features, found {features.shape[1]}."
        )

    values = features.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise DataValidationError("Feature matrix contains infinite values.")
    missing_by_column = features.isna().mean()
    too_sparse = missing_by_column[missing_by_column > max_missing_fraction]
    if not too_sparse.empty:
        raise DataValidationError(
            f"Columns exceed max_missing_fraction={max_missing_fraction}: {list(too_sparse.index)}"
        )

    labels_float = target.to_numpy(dtype=np.float64)
    if not np.all(np.mod(labels_float, 1) == 0):
        raise DataValidationError("Target values must be integers.")
    labels = labels_float.astype(np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise DataValidationError("Target must contain at least two classes.")
    if not np.array_equal(classes, np.arange(len(classes))):
        raise DataValidationError(f"Class labels must be contiguous 0..K-1, got {classes.tolist()}.")
    if counts.min() < MIN_PER_CLASS:
        raise DataValidationError(f"Each class needs at least {MIN_PER_CLASS} samples.")

    return DataReport(
        n_rows=len(frame),
        n_features=features.shape[1],
        n_classes=len(classes),
        class_counts={int(c): int(n) for c, n in zip(classes, counts)},
        missing_fraction=float(np.isnan(values).mean()),
        columns_with_missing=[str(c) for c in features.columns[features.isna().any()]],
        feature_names=[str(c) for c in features.columns],
    )


def validate_feature_matrix(
    features: Any,
    expected_n_features: int,
    name: str = "features",
    allow_nan: bool = True,
) -> np.ndarray:
    try:
        array = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be numeric: {exc}") from exc
    if array.ndim != 2:
        raise DataValidationError(f"{name} must be 2-D (rows, features); got ndim={array.ndim}.")
    if array.shape[0] == 0:
        raise DataValidationError(f"{name} contains no rows.")
    if array.shape[1] != expected_n_features:
        raise DataValidationError(
            f"{name} has {array.shape[1]} features per row; expected {expected_n_features}."
        )
    if np.isinf(array).any():
        raise DataValidationError(f"{name} contains infinite values.")
    if not allow_nan and np.isnan(array).any():
        raise DataValidationError(f"{name} contains NaN values.")
    return array