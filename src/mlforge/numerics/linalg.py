from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mlforge.exceptions import DataValidationError
from mlforge.numerics.stability import ToleranceReport, compare_arrays


def _check_matmul_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise DataValidationError(f"matmul needs 2-D inputs; got {a.ndim}-D and {b.ndim}-D.")
    if a.shape[1] != b.shape[0]:
        raise DataValidationError(f"Inner dimensions differ: {a.shape} @ {b.shape}.")


def matmul_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    _check_matmul_shapes(a, b)
    m, k = a.shape
    n = b.shape[1]
    out = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a[i, p] * b[p, j]
            out[i, j] = acc
    return out


def matmul_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    _check_matmul_shapes(a, b)
    return a @ b


def verify_matmul(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-9, atol: float = 1e-12
) -> ToleranceReport:
    return compare_arrays(matmul_vectorized(a, b), matmul_reference(a, b), rtol=rtol, atol=atol)


def manual_linear_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    out = x @ weight.T
    return out if bias is None else out + bias


@torch.no_grad()
def verify_linear_layer(layer: nn.Linear, x: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-6) -> ToleranceReport:
    return compare_arrays(layer(x), manual_linear_forward(x, layer.weight, layer.bias), rtol=rtol, atol=atol)


def standardize_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(std < eps, 1.0, std)


def pairwise_squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise DataValidationError(f"Need 2-D inputs with equal feature dim; got {a.shape}, {b.shape}.")
    a_sq = (a**2).sum(axis=1, keepdims=True)
    b_sq = (b**2).sum(axis=1)[None, :]
    return np.maximum(a_sq + b_sq - 2.0 * (a @ b.T), 0.0)


def pairwise_squared_distances_broadcast(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    return (diff**2).sum(axis=-1)