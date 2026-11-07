import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mlforge.config import BaselineConfig
from mlforge.exceptions import ConfigError, DataValidationError
from mlforge.memory.profiler import count_parameters
from mlforge.models.baseline import LogisticRegressionBaseline
from mlforge.models.mlp import ACTIVATIONS, MLP


def test_forward_output_shape() -> None:
    model = MLP(5, [8, 4], 3)
    assert model(torch.randn(7, 5)).shape == (7, 3)


def test_parameter_count_matches_formula() -> None:
    model = MLP(4, [8], 2)
    assert count_parameters(model) == (4 * 8 + 8) + (8 * 2 + 2)


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_all_activations_run(activation: str) -> None:
    out = MLP(3, [4], 2, activation=activation)(torch.randn(2, 3))
    assert torch.isfinite(out).all()


def test_invalid_arguments_rejected() -> None:
    with pytest.raises(ConfigError):
        MLP(4, [8], 2, activation="nope")
    with pytest.raises(ConfigError):
        MLP(0, [8], 2)
    with pytest.raises(ConfigError):
        MLP(4, [0], 2)
    with pytest.raises(ConfigError):
        MLP(4, [8], 2, dropout=1.0)


def test_input_shape_validation() -> None:
    model = MLP(4, [8], 2)
    with pytest.raises(DataValidationError):
        model(torch.randn(3, 5))
    with pytest.raises(DataValidationError):
        model(torch.randn(4))


def test_loss_matches_functional_cross_entropy() -> None:
    logits = torch.randn(6, 3)
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    assert torch.isclose(MLP.compute_loss(logits, targets), F.cross_entropy(logits, targets))


def test_eval_mode_is_deterministic_with_dropout() -> None:
    model = MLP(4, [16], 2, dropout=0.5).eval()
    x = torch.randn(5, 4)
    assert torch.equal(model(x), model(x))


def test_init_config_roundtrip_reproduces_outputs() -> None:
    model = MLP(4, [8, 4], 2, activation="tanh").eval()
    clone = MLP.from_init_config(model.init_config()).eval()
    clone.load_state_dict(model.state_dict())
    x = torch.randn(3, 4)
    assert torch.equal(model(x), clone(x))


def test_baseline_fit_predict() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] > 0).astype(int)
    baseline = LogisticRegressionBaseline(BaselineConfig(), seed=0).fit(x, y)
    proba = baseline.predict_proba(x)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert (baseline.predict(x) == y).mean() > 0.9


def test_baseline_requires_fit() -> None:
    with pytest.raises(DataValidationError, match="not fitted"):
        LogisticRegressionBaseline(BaselineConfig(), seed=0).predict_proba(np.zeros((1, 2)))