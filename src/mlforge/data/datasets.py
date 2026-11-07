from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from mlforge.config import DataConfig
from mlforge.exceptions import DataValidationError
from mlforge.logging_utils import get_logger

logger = get_logger(__name__)
FEATURE_PREFIX = "feature_"


def generate_synthetic_dataset(cfg: DataConfig, seed: int) -> pd.DataFrame:
    features, labels = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_classes=cfg.n_classes,
        class_sep=cfg.class_sep,
        flip_y=cfg.flip_y,
        random_state=seed,
    )
    columns = [f"{FEATURE_PREFIX}{i}" for i in range(cfg.n_features)]
    frame = pd.DataFrame(features, columns=columns)
    if cfg.missing_fraction > 0:
        rng = np.random.default_rng(seed + 1)
        mask = rng.random(features.shape) < cfg.missing_fraction
        frame = frame.mask(mask)
    frame[cfg.target_column] = labels.astype(np.int64)
    logger.info("Generated synthetic dataset: %d rows x %d features.", *frame.shape)
    return frame


def load_csv_dataset(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise DataValidationError(f"CSV file not found: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not parse CSV '{csv_path}': {exc}") from exc
    logger.info("Loaded CSV dataset %s with shape %s.", csv_path, frame.shape)
    return frame


def load_dataset(cfg: DataConfig, seed: int) -> pd.DataFrame:
    if cfg.csv_path:
        return load_csv_dataset(cfg.csv_path)
    return generate_synthetic_dataset(cfg, seed)