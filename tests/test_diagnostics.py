import pytest
import torch

from mlforge.checkpointing import load_checkpoint
from mlforge.diagnostics import (
    check_loss_finite,
    check_model_device,
    check_model_outputs,
    check_tensor,
    inspect_gradients,
    is_oom_error,
    require_checkpoint,
)
from mlforge.exceptions import (
    CheckpointError,
    DataValidationError,
    DeviceError,
    NumericalInstabilityError,
)
from mlforge.models.mlp import MLP

EXPLODE = 1000.0
VANISH = 1e-7


def test_check_loss_finite() -> None:
    assert check_loss_finite(torch.tensor(1.5), "ctx") == pytest.approx(1.5)
    with pytest.raises(NumericalInstabilityError, match="NaN loss"):
        check_loss_finite(float("nan"), "ctx")
    with pytest.raises(NumericalInstabilityError, match="Infinite loss"):
        check_loss_finite(float("inf"), "ctx")


def test_check_model_outputs() -> None:
    check_model_outputs(torch.zeros(3, 2), 2)
    with pytest.raises(DataValidationError, match="output shape"):
        check_model_outputs(torch.zeros(3, 4), 2)
    with pytest.raises(NumericalInstabilityError):
        check_model_outputs(torch.tensor([[float("nan"), 0.0]]), 2)


def test_check_tensor_variants() -> None:
    tensor = torch.zeros(2, 3)
    check_tensor(tensor, "x", ndim=2, last_dim=3, dtype=torch.float32, device=torch.device("cpu"))
    with pytest.raises(DataValidationError, match="dims"):
        check_tensor(tensor, "x", ndim=3)
    with pytest.raises(DataValidationError, match="last dim"):
        check_tensor(tensor, "x", last_dim=4)
    with pytest.raises(DataValidationError, match="dtype"):
        check_tensor(tensor, "x", dtype=torch.float64)
    with pytest.raises(DeviceError):
        check_tensor(tensor, "x", device=torch.device("cuda"))
    with pytest.raises(NumericalInstabilityError):
        check_tensor(torch.tensor([float("inf")]), "x")
    with pytest.raises(DataValidationError, match="torch.Tensor"):
        check_tensor([1, 2], "x")  # type: ignore[arg-type]


def test_check_model_device_mismatch() -> None:
    model = MLP(4, [4], 2)
    check_model_device(model, torch.zeros(1, 4))
    with pytest.raises(DeviceError, match="mismatch"):
        check_model_device(model, torch.empty(1, 4, device="meta"))


def _backward(model: MLP) -> None:
    loss = MLP.compute_loss(model(torch.randn(16, 4)), torch.randint(0, 2, (16,)))
    loss.backward()


def test_inspect_gradients_healthy_and_missing() -> None:
    model = MLP(4, [6], 2)
    before = inspect_gradients(model, EXPLODE, VANISH)
    assert len(before.layers_without_grad) == 4 and before.has_issues
    _backward(model)
    after = inspect_gradients(model, EXPLODE, VANISH)
    assert after.global_norm > 0 and not after.nonfinite_layers and not after.exploding_layers


def test_inspect_gradients_flags_vanishing_exploding_nonfinite() -> None:
    model = MLP(4, [6], 2)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    assert len(inspect_gradients(model, EXPLODE, VANISH).vanishing_layers) == 4
    for p in model.parameters():
        p.grad = torch.full_like(p, 1e6)
    assert len(inspect_gradients(model, EXPLODE, VANISH).exploding_layers) == 4
    next(model.parameters()).grad = torch.full_like(next(model.parameters()), float("nan"))
    report = inspect_gradients(model, EXPLODE, VANISH)
    assert report.nonfinite_layers and report.global_norm != report.global_norm


def test_missing_checkpoint_raises(tmp_path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        require_checkpoint(tmp_path / "missing.pt")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "missing.pt")


def test_is_oom_error() -> None:
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert is_oom_error(MemoryError())
    assert not is_oom_error(RuntimeError("shape mismatch"))
    assert not is_oom_error(ValueError("out of memory"))