from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlforge.config import ModelConfig
from mlforge.exceptions import ConfigError, DataValidationError

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}
_KAIMING_ACTIVATIONS = {"relu", "leaky_relu"}


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 2:
            raise ConfigError("MLP requires input_dim >= 1 and output_dim >= 2.")
        if activation not in ACTIVATIONS:
            raise ConfigError(f"Unknown activation '{activation}'. Options: {sorted(ACTIVATIONS)}")
        if not 0.0 <= dropout < 1.0:
            raise ConfigError("dropout must be in [0, 1).")
        hidden = tuple(int(h) for h in hidden_dims)
        if any(h < 1 for h in hidden):
            raise ConfigError("hidden_dims must be positive integers.")

        self.input_dim = input_dim
        self.hidden_dims = hidden
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout_p = dropout

        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.append(nn.Linear(previous, width))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)
        self._initialise_weights()

    def _initialise_weights(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                if self.activation_name in _KAIMING_ACTIVATIONS:
                    nn.init.kaiming_normal_(module.weight, nonlinearity=self.activation_name)
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise DataValidationError(
                f"MLP expects input of shape (batch, {self.input_dim}); got {tuple(x.shape)}."
            )
        return self.network(x)

    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)

    def init_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "output_dim": self.output_dim,
            "activation": self.activation_name,
            "dropout": self.dropout_p,
        }

    @classmethod
    def from_init_config(cls, config: dict[str, Any]) -> MLP:
        try:
            return cls(**config)
        except TypeError as exc:
            raise ConfigError(f"Invalid MLP init config: {exc}") from exc


def build_mlp(cfg: ModelConfig, input_dim: int, output_dim: int) -> MLP:
    return MLP(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        output_dim=output_dim,
        activation=cfg.activation,
        dropout=cfg.dropout,
    )