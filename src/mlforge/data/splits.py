from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> DatasetSplits:
    x_rest, x_test, y_rest, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    relative_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest, y_rest, test_size=relative_val, random_state=seed, stratify=y_rest
    )
    return DatasetSplits(x_train, x_val, x_test, y_train, y_val, y_test)