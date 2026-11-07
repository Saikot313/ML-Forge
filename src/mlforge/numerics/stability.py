from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from mlforge.exceptions import DataValidationError, NumericalInstabilityError

DEFAULT_NORM_EPS = 1e-12


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def naive_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    max_val = np.max(x, axis=axis, keepdims=True)
    out = max_val + np.log(np.exp(x - max_val).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def naive_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_neg = np.exp(values[~positive])
    out[~positive] = exp_neg / (1.0 + exp_neg)
    return out


def safe_normalize(x: np.ndarray, axis: int = -1, eps: float = DEFAULT_NORM_EPS) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

def stable_log_softmax_t(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    return shifted - torch.logsumexp(shifted, dim=dim, keepdim=True)


def cross_entropy_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise DataValidationError(
            f"Expected logits (N, C) and targets (N,); got {tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    log_probs = stable_log_softmax_t(logits, dim=1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise DataValidationError(f"Unknown reduction '{reduction}'.")


def safe_normalize_t(x: torch.Tensor, dim: int = -1, eps: float = DEFAULT_NORM_EPS) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)



def count_nonfinite(value: Any) -> dict[str, int]:
    array = to_numpy(value)
    if not np.issubdtype(array.dtype, np.floating):
        return {"nan": 0, "inf": 0}
    return {"nan": int(np.isnan(array).sum()), "inf": int(np.isinf(array).sum())}


def assert_finite(value: Any, name: str = "value") -> None:
    counts = count_nonfinite(value)
    if counts["nan"] or counts["inf"]:
        raise NumericalInstabilityError(
            f"{name} contains {counts['nan']} NaN and {counts['inf']} Inf entries."
        )

@dataclass(frozen=True)
class ToleranceReport:
    allclose: bool
    max_abs_diff: float
    max_rel_diff: float
    n_mismatched: int
    n_elements: int
    rtol: float
    atol: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_arrays(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> ToleranceReport:
    left = to_numpy(a).astype(np.float64)
    right = to_numpy(b).astype(np.float64)
    if left.shape != right.shape:
        raise DataValidationError(f"Shape mismatch: {left.shape} vs {right.shape}.")
    close = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=False)
    abs_diff = np.abs(left - right)
    denom = np.maximum(np.abs(right), np.finfo(np.float64).tiny)
    return ToleranceReport(
        allclose=bool(close.all()),
        max_abs_diff=float(np.nanmax(abs_diff)) if abs_diff.size else 0.0,
        max_rel_diff=float(np.nanmax(abs_diff / denom)) if abs_diff.size else 0.0,
        n_mismatched=int((~close).sum()),
        n_elements=int(close.size),
        rtol=rtol,
        atol=atol,
    )