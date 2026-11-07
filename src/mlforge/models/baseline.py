from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlforge.config import BaselineConfig
from mlforge.exceptions import DataValidationError
from mlforge.logging_utils import get_logger

logger = get_logger(__name__)


class LogisticRegressionBaseline:
    def __init__(self, config: BaselineConfig, seed: int) -> None:
        self.model = LogisticRegression(C=config.C, max_iter=config.max_iter, random_state=seed)
        self.max_iter = config.max_iter
        self.n_iter: int | None = None
        self._fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LogisticRegressionBaseline:
        self.model.fit(features, labels)
        if not np.array_equal(self.model.classes_, np.arange(len(self.model.classes_))):
            raise DataValidationError("Baseline requires class labels 0..K-1.")
        self.n_iter = int(np.max(self.model.n_iter_))
        if self.n_iter >= self.max_iter:
            logger.warning("Logistic Regression reached max_iter=%d; it may not have converged.", self.max_iter)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise DataValidationError("Baseline model is not fitted.")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.model.predict_proba(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self.model.predict(features)