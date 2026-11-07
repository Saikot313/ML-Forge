from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from mlforge.exceptions import DataValidationError

DEFAULT_EPS = 1e-8  # std below this is treated as constant -> scale 1.0


class TabularPreprocessor:
    def __init__(self, eps: float = DEFAULT_EPS) -> None:
        self.eps = eps
        self.fill_values: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.fill_values is not None

    @property
    def n_features(self) -> int:
        self._require_fitted()
        assert self.mean is not None
        return int(self.mean.shape[0])

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise DataValidationError("TabularPreprocessor is not fitted.")

    def fit(self, features: np.ndarray) -> TabularPreprocessor:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0:
            raise DataValidationError("fit() requires a non-empty 2-D array.")
        if np.isinf(array).any():
            raise DataValidationError("fit() received infinite values.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN column handled below
            fill = np.nanmedian(array, axis=0)
        if np.isnan(fill).any():
            bad = np.flatnonzero(np.isnan(fill)).tolist()
            raise DataValidationError(f"Columns {bad} contain only missing values.")
        imputed = np.where(np.isnan(array), fill, array)
        std = imputed.std(axis=0)
        self.fill_values = fill
        self.mean = imputed.mean(axis=0)
        self.scale = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.fill_values is not None and self.mean is not None and self.scale is not None
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.n_features:
            raise DataValidationError(
                f"transform() expects shape (n, {self.n_features}); got {array.shape}."
            )
        if np.isinf(array).any():
            raise DataValidationError("transform() received infinite values.")
        imputed = np.where(np.isnan(array), self.fill_values, array)
        return np.ascontiguousarray((imputed - self.mean) / self.scale, dtype=np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.fill_values is not None and self.mean is not None and self.scale is not None
        return {
            "eps": self.eps,
            "fill_values": self.fill_values.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> TabularPreprocessor:
        try:
            instance = cls(eps=float(state["eps"]))
            instance.fill_values = np.asarray(state["fill_values"], dtype=np.float64)
            instance.mean = np.asarray(state["mean"], dtype=np.float64)
            instance.scale = np.asarray(state["scale"], dtype=np.float64)
        except KeyError as exc:
            raise DataValidationError(f"Preprocessor state missing key: {exc}") from exc
        return instance