from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from mlforge.exceptions import (
    CheckpointError,
    DataValidationError,
    DeviceError,
    NumericalInstabilityError,
)
from mlforge.logging_utils import get_logger
from mlforge.numerics.stability import count_nonfinite

logger = get_logger(__name__)


def check_loss_finite(loss: torch.Tensor | float, context: str) -> float:
    value = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
    if math.isnan(value):
        raise NumericalInstabilityError(
            f"NaN loss at {context}. Likely causes: learning rate too high, invalid inputs "
            "(NaN/Inf features), log(0) in a custom loss, or exploding gradients."
        )
    if math.isinf(value):
        raise NumericalInstabilityError(
            f"Infinite loss at {context}. Likely causes: overflow in logits/exp, "
            "extreme feature scales, or a diverging optimisation."
        )
    return value


def check_model_outputs(logits: torch.Tensor, expected_classes: int) -> None:
    if logits.ndim != 2 or logits.shape[1] != expected_classes:
        raise DataValidationError(
            f"Invalid model output shape {tuple(logits.shape)}; expected (N, {expected_classes})."
        )
    counts = count_nonfinite(logits)
    if counts["nan"] or counts["inf"]:
        raise NumericalInstabilityError(
            f"Model output has {counts['nan']} NaN and {counts['inf']} Inf values "
            "(check input scaling, mixed precision overflow and weights)."
        )


def _same_device(a: torch.device, b: torch.device) -> bool:
    if a.type != b.type:
        return False
    return a.index is None or b.index is None or a.index == b.index


def check_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    ndim: int | None = None,
    last_dim: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    require_finite: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise DataValidationError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if ndim is not None and tensor.ndim != ndim:
        raise DataValidationError(f"{name} must have {ndim} dims; got shape {tuple(tensor.shape)}.")
    if last_dim is not None and (tensor.ndim == 0 or tensor.shape[-1] != last_dim):
        raise DataValidationError(f"{name} last dim must be {last_dim}; got shape {tuple(tensor.shape)}.")
    if dtype is not None and tensor.dtype != dtype:
        raise DataValidationError(f"{name} dtype mismatch: expected {dtype}, got {tensor.dtype}.")
    if device is not None and not _same_device(tensor.device, device):
        raise DeviceError(f"{name} is on {tensor.device} but {device} was expected.")
    if require_finite and tensor.is_floating_point():
        counts = count_nonfinite(tensor)
        if counts["nan"] or counts["inf"]:
            raise NumericalInstabilityError(
                f"{name} contains {counts['nan']} NaN and {counts['inf']} Inf values."
            )


def check_model_device(model: nn.Module, *tensors: torch.Tensor) -> None:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        return
    for index, tensor in enumerate(tensors):
        if not _same_device(tensor.device, model_device):
            raise DeviceError(
                f"Device mismatch: model on {model_device}, tensor #{index} on {tensor.device}."
            )


@dataclass
class GradientReport:
    global_norm: float
    layer_norms: dict[str, float] = field(default_factory=dict)
    nonfinite_layers: list[str] = field(default_factory=list)
    exploding_layers: list[str] = field(default_factory=list)
    vanishing_layers: list[str] = field(default_factory=list)
    layers_without_grad: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.nonfinite_layers
            or self.exploding_layers
            or self.vanishing_layers
            or self.layers_without_grad
        )

    def summary(self) -> str:
        return (
            f"global_norm={self.global_norm:.4g}, nonfinite={self.nonfinite_layers}, "
            f"exploding={self.exploding_layers}, vanishing={self.vanishing_layers}, "
            f"no_grad={self.layers_without_grad}"
        )


def inspect_gradients(
    model: nn.Module, explode_threshold: float, vanish_threshold: float
) -> GradientReport:
    squared_sum = 0.0
    report = GradientReport(global_norm=0.0)
    for name, param in model.named_parameters():
        if param.grad is None:
            report.layers_without_grad.append(name)
            continue
        norm = float(torch.linalg.vector_norm(param.grad.detach()))
        report.layer_norms[name] = norm
        if not math.isfinite(norm):
            report.nonfinite_layers.append(name)
            squared_sum = float("nan")
            continue
        if norm > explode_threshold:
            report.exploding_layers.append(name)
        elif norm < vanish_threshold:
            report.vanishing_layers.append(name)
        squared_sum += norm**2
    report.global_norm = math.sqrt(squared_sum) if math.isfinite(squared_sum) else float("nan")
    return report


def require_checkpoint(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise CheckpointError(
            f"Checkpoint not found: {checkpoint_path}. Run training first or check the path."
        )
    return checkpoint_path


def is_oom_error(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    if isinstance(exc, MemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()