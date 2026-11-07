import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mlforge.exceptions import DataValidationError, NumericalInstabilityError
from mlforge.numerics import (
    assert_finite,
    clip_gradients,
    compare_arrays,
    count_nonfinite,
    cross_entropy_from_logits,
    gradient_norm,
    logsumexp,
    naive_sigmoid,
    naive_softmax,
    pairwise_squared_distances,
    pairwise_squared_distances_broadcast,
    safe_normalize,
    stable_log_softmax_t,
    stable_sigmoid,
    stable_softmax,
    standardize_columns,
    verify_linear_layer,
    verify_matmul,
)


def test_naive_softmax_overflows_but_stable_does_not() -> None:
    x = np.array([[1000.0, 1001.0]])
    with np.errstate(all="ignore"):
        assert np.isnan(naive_softmax(x)).any()
    stable = stable_softmax(x)
    np.testing.assert_allclose(stable, [[0.2689414, 0.7310586]], rtol=1e-6)
    assert np.isclose(stable.sum(), 1.0)


def test_stable_softmax_matches_torch() -> None:
    x = np.random.default_rng(0).normal(size=(5, 4))
    expected = torch.softmax(torch.from_numpy(x), dim=-1)
    assert compare_arrays(stable_softmax(x), expected, rtol=1e-9, atol=1e-12).allclose


def test_logsumexp_large_inputs() -> None:
    np.testing.assert_allclose(logsumexp(np.array([1000.0, 1000.0])), 1000.0 + np.log(2.0))


def test_stable_sigmoid_avoids_overflow_warning() -> None:
    with pytest.warns(RuntimeWarning):
        naive_sigmoid(np.array([-1000.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = stable_sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-12)
    x = np.linspace(-5, 5, 11)
    np.testing.assert_allclose(stable_sigmoid(-x), 1.0 - stable_sigmoid(x), atol=1e-12)


def test_safe_normalize_handles_zero_vectors() -> None:
    out = safe_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out[0], [0.6, 0.8])
    assert np.all(out[1] == 0.0) and not np.isnan(out).any()


def test_stable_log_softmax_and_cross_entropy_match_torch() -> None:
    logits = torch.randn(6, 4)
    targets = torch.tensor([0, 1, 2, 3, 0, 1])
    assert torch.allclose(stable_log_softmax_t(logits), torch.log_softmax(logits, dim=-1), atol=1e-6)
    assert torch.isclose(cross_entropy_from_logits(logits, targets), F.cross_entropy(logits, targets))
    assert cross_entropy_from_logits(logits, targets, reduction="none").shape == (6,)


def test_cross_entropy_finite_for_extreme_logits() -> None:
    logits = torch.tensor([[1e4, -1e4], [-1e4, 1e4]])
    loss = cross_entropy_from_logits(logits, torch.tensor([0, 1]))
    assert torch.isfinite(loss) and loss.item() == pytest.approx(0.0, abs=1e-6)


def test_cross_entropy_shape_validation() -> None:
    with pytest.raises(DataValidationError):
        cross_entropy_from_logits(torch.randn(3, 2), torch.tensor([0, 1]))


def test_nonfinite_detection() -> None:
    arr = np.array([1.0, np.nan, np.inf, -np.inf])
    assert count_nonfinite(arr) == {"nan": 1, "inf": 2}
    assert count_nonfinite(torch.tensor([float("nan")])) == {"nan": 1, "inf": 0}
    assert count_nonfinite(np.array([1, 2])) == {"nan": 0, "inf": 0}
    assert_finite(np.ones(3))
    with pytest.raises(NumericalInstabilityError, match="NaN"):
        assert_finite(arr, "arr")


def test_compare_arrays() -> None:
    assert compare_arrays([1.0, 2.0], [1.0, 2.0 + 1e-9], rtol=1e-6, atol=0).allclose
    report = compare_arrays([1.0, np.nan], [1.0, np.nan])
    assert not report.allclose and report.n_mismatched == 1
    with pytest.raises(DataValidationError, match="Shape mismatch"):
        compare_arrays(np.zeros(2), np.zeros(3))


def test_matmul_vectorised_matches_reference() -> None:
    rng = np.random.default_rng(0)
    assert verify_matmul(rng.normal(size=(4, 5)), rng.normal(size=(5, 3))).allclose
    with pytest.raises(DataValidationError):
        verify_matmul(np.zeros((2, 3)), np.zeros((4, 2)))


def test_manual_linear_matches_nn_linear() -> None:
    layer = torch.nn.Linear(5, 3)
    assert verify_linear_layer(layer, torch.randn(4, 5)).allclose


def test_pairwise_distances_match_broadcast_and_nonnegative() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(6, 4)), rng.normal(size=(5, 4))
    fast = pairwise_squared_distances(a, b)
    np.testing.assert_allclose(fast, pairwise_squared_distances_broadcast(a, b), atol=1e-9)
    assert (fast >= 0).all()


def test_standardize_columns_constant_column() -> None:
    out = standardize_columns(np.column_stack([np.ones(5), np.arange(5.0)]))
    assert np.all(out[:, 0] == 0.0) and out[:, 1].std() == pytest.approx(1.0)


def test_gradient_norm_and_clipping() -> None:
    param = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    param.grad = torch.tensor([3.0, 4.0])
    assert gradient_norm([param]) == pytest.approx(5.0)
    assert clip_gradients([param], max_norm=10.0) == pytest.approx(5.0)
    assert gradient_norm([param]) == pytest.approx(5.0)  # unchanged below threshold
    assert clip_gradients([param], max_norm=1.0) == pytest.approx(5.0)
    assert gradient_norm([param]) == pytest.approx(1.0, abs=1e-4)


def test_clipping_matches_torch_reference() -> None:
    torch.manual_seed(0)
    ours = [torch.nn.Parameter(torch.randn(5, 3)), torch.nn.Parameter(torch.randn(3))]
    for p in ours:
        p.grad = torch.randn_like(p) * 10
    ref = [torch.nn.Parameter(p.detach().clone()) for p in ours]
    for a, b in zip(ours, ref):
        b.grad = a.grad.clone()
    expected_norm = gradient_norm(ref)
    assert clip_gradients(ours, 1.0) == pytest.approx(expected_norm, rel=1e-6)
    torch.nn.utils.clip_grad_norm_(ref, 1.0)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)


def test_clipping_rejects_nonfinite_gradients_and_bad_args() -> None:
    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = torch.tensor([float("nan"), 1.0])
    with pytest.raises(NumericalInstabilityError):
        clip_gradients([param], 1.0)
    with pytest.raises(ValueError):
        clip_gradients([param], 0.0)
    assert gradient_norm([torch.nn.Parameter(torch.zeros(1))]) == 0.0