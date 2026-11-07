import numpy as np
import pandas as pd
import pytest

from mlforge.config import DataConfig, ExperimentConfig
from mlforge.data.datasets import generate_synthetic_dataset, load_csv_dataset
from mlforge.data.pipeline import prepare_data
from mlforge.data.preprocessing import TabularPreprocessor
from mlforge.data.splits import split_dataset
from mlforge.data.validation import validate_dataframe, validate_feature_matrix
from mlforge.exceptions import DataValidationError

CFG = DataConfig(n_samples=200, n_features=6, n_informative=4, n_redundant=1)


def make_frame() -> pd.DataFrame:
    return generate_synthetic_dataset(CFG, seed=1)


def test_synthetic_generation_is_repeatable() -> None:
    pd.testing.assert_frame_equal(make_frame(), generate_synthetic_dataset(CFG, seed=1))
    assert not make_frame().equals(generate_synthetic_dataset(CFG, seed=2))


def test_validate_dataframe_report() -> None:
    report = validate_dataframe(make_frame(), "target")
    assert report.n_rows == 200 and report.n_features == 6 and report.n_classes == 2
    assert sum(report.class_counts.values()) == 200
    assert 0 < report.missing_fraction < 0.1


def test_validate_rejects_bad_tables() -> None:
    frame = make_frame()
    with pytest.raises(DataValidationError, match="not found"):
        validate_dataframe(frame, "label")
    with pytest.raises(DataValidationError, match="empty"):
        validate_dataframe(frame.iloc[0:0], "target")
    with pytest.raises(DataValidationError, match="Non-numeric"):
        validate_dataframe(frame.assign(feature_0="x"), "target")
    inf_frame = frame.copy()
    inf_frame.loc[0, "feature_1"] = np.inf
    with pytest.raises(DataValidationError, match="infinite"):
        validate_dataframe(inf_frame, "target")
    with pytest.raises(DataValidationError, match="two classes"):
        validate_dataframe(frame.assign(target=0), "target")
    sparse = frame.copy()
    sparse["feature_2"] = np.nan
    with pytest.raises(DataValidationError, match="max_missing_fraction"):
        validate_dataframe(sparse, "target", max_missing_fraction=0.5)
    with pytest.raises(DataValidationError, match="Expected 3 features"):
        validate_dataframe(frame, "target", expected_n_features=3)


def test_validate_feature_matrix() -> None:
    ok = validate_feature_matrix([[1.0, None, 3.0]], expected_n_features=3)
    assert ok.shape == (1, 3) and np.isnan(ok[0, 1])
    with pytest.raises(DataValidationError, match="2-D"):
        validate_feature_matrix([1.0, 2.0], expected_n_features=2)
    with pytest.raises(DataValidationError, match="expected 3"):
        validate_feature_matrix([[1.0, 2.0]], expected_n_features=3)
    with pytest.raises(DataValidationError, match="infinite"):
        validate_feature_matrix([[np.inf, 1.0]], expected_n_features=2)
    with pytest.raises(DataValidationError, match="NaN"):
        validate_feature_matrix([[np.nan, 1.0]], expected_n_features=2, allow_nan=False)
    with pytest.raises(DataValidationError, match="numeric"):
        validate_feature_matrix([["a", "b"]], expected_n_features=2)


def test_preprocessor_standardises_and_imputes() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 3.0, size=(500, 4))
    data[::10, 1] = np.nan
    pre = TabularPreprocessor().fit(data)
    out = pre.transform(data)
    assert out.dtype == np.float32 and not np.isnan(out).any()
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=0.05)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=0.05)


def test_preprocessor_constant_column_and_errors() -> None:
    data = np.column_stack([np.ones(20), np.arange(20, dtype=float)])
    out = TabularPreprocessor().fit_transform(data)
    assert np.all(out[:, 0] == 0.0)
    with pytest.raises(DataValidationError, match="not fitted"):
        TabularPreprocessor().transform(data)
    with pytest.raises(DataValidationError, match="only missing"):
        TabularPreprocessor().fit(np.full((5, 2), np.nan))
    with pytest.raises(DataValidationError, match="expects shape"):
        TabularPreprocessor().fit(data).transform(np.ones((3, 5)))


def test_preprocessor_state_roundtrip() -> None:
    data = np.random.default_rng(1).normal(size=(50, 3))
    pre = TabularPreprocessor().fit(data)
    restored = TabularPreprocessor.from_dict(pre.to_dict())
    np.testing.assert_array_equal(pre.transform(data), restored.transform(data))


def test_split_is_reproducible_disjoint_and_stratified() -> None:
    features = np.arange(400, dtype=float).reshape(200, 2)
    labels = np.array([0, 1] * 100)
    a = split_dataset(features, labels, 0.15, 0.15, seed=3)
    b = split_dataset(features, labels, 0.15, 0.15, seed=3)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    assert len(a.y_train) + len(a.y_val) + len(a.y_test) == 200
    train_ids, val_ids, test_ids = (set(s[:, 0]) for s in (a.X_train, a.X_val, a.X_test))
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
    assert abs(a.y_test.mean() - 0.5) < 0.1


def test_csv_loading(tmp_path) -> None:
    path = tmp_path / "data.csv"
    make_frame().to_csv(path, index=False)
    assert load_csv_dataset(path).shape == (200, 7)
    with pytest.raises(DataValidationError, match="not found"):
        load_csv_dataset(tmp_path / "missing.csv")


def test_prepare_data_outputs(quick_cfg: ExperimentConfig) -> None:
    prepared = prepare_data(quick_cfg.data, quick_cfg.seed)
    s = prepared.splits
    assert len(s.y_train) + len(s.y_val) + len(s.y_test) == quick_cfg.data.n_samples
    assert s.X_train.dtype == np.float32 and s.X_train.shape[1] == prepared.n_features == 8
    assert not np.isnan(s.X_train).any() and not np.isnan(s.X_test).any()
    assert prepared.preprocessor.n_features == 8