from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from mlforge.exceptions import NumericalInstabilityError

DEFAULT_CLIP_EPS = 1e-6


def gradient_norm(parameters: Iterable[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    per_tensor = torch.stack([torch.linalg.vector_norm(g, norm_type) for g in grads])
    return float(torch.linalg.vector_norm(per_tensor, norm_type))


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = DEFAULT_CLIP_EPS,
) -> float:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")
    params = list(parameters)
    total = gradient_norm(params)
    if not math.isfinite(total):
        raise NumericalInstabilityError(
            f"Gradient norm is {total}; gradients are non-finite and cannot be clipped."
        )
    if total > max_norm:
        scale = max_norm / (total + eps)
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scale)
    return total