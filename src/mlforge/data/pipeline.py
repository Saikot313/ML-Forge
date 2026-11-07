from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlforge.config import DataConfig
from mlforge.data.datasets import load_dataset
from mlforge.data.preprocessing import TabularPreprocessor
from mlforge.data.splits import DatasetSplits, split_dataset
from mlforge.data.validation import DataReport, validate_dataframe
from mlforge.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreparedData:
    splits: DatasetSplits
    preprocessor: TabularPreprocessor
    feature_names: list[str]
    n_features: int
    n_classes: int
    report: DataReport


def prepare_data(cfg: DataConfig, seed: int) -> PreparedData:
    frame = load_dataset(cfg, seed)
    report = validate_dataframe(
        frame, cfg.target_column, max_missing_fraction=cfg.max_missing_fraction
    )
    features = frame[report.feature_names].to_numpy(dtype=np.float64)
    labels = frame[cfg.target_column].to_numpy(dtype=np.int64)

    raw = split_dataset(features, labels, cfg.val_size, cfg.test_size, seed)
    preprocessor = TabularPreprocessor().fit(raw.X_train)
    splits = DatasetSplits(
        X_train=preprocessor.transform(raw.X_train),
        X_val=preprocessor.transform(raw.X_val),
        X_test=preprocessor.transform(raw.X_test),
        y_train=raw.y_train,
        y_val=raw.y_val,
        y_test=raw.y_test,
    )
    logger.info(
        "Prepared data: train=%d val=%d test=%d features=%d classes=%d",
        len(splits.y_train),
        len(splits.y_val),
        len(splits.y_test),
        report.n_features,
        report.n_classes,
    )
    return PreparedData(
        splits=splits,
        preprocessor=preprocessor,
        feature_names=report.feature_names,
        n_features=report.n_features,
        n_classes=report.n_classes,
        report=report,
    )