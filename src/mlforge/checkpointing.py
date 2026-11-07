from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlforge.diagnostics.checks import require_checkpoint
from mlforge.exceptions import CheckpointError
from mlforge.logging_utils import get_logger
from mlforge.models.mlp import MLP

logger = get_logger(__name__)
CHECKPOINT_VERSION = 1
_REQUIRED_KEYS = ("version", "model_state_dict", "model_init_config", "preprocessor", "feature_names")


@dataclass
class CheckpointContext:
    config: dict[str, Any]
    preprocessor_state: dict[str, Any]
    feature_names: list[str]
    seed: int


def _to_plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=float))


def build_metadata(seed: int, device: torch.device | str) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": seed,
        "device": str(device),
    }


def save_checkpoint(
    path: str | Path,
    model: MLP,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_val_loss: float,
    history: dict[str, Any],
    context: CheckpointContext,
    metrics: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_init_config": _to_plain(model.init_config()),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "history": _to_plain(history),
        "config": _to_plain(context.config),
        "preprocessor": _to_plain(context.preprocessor_state),
        "feature_names": list(context.feature_names),
        "metrics": _to_plain(metrics) if metrics is not None else None,
        "metadata": build_metadata(context.seed, device),
    }
    temp_path = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, target)
    except OSError as exc:
        raise CheckpointError(f"Failed to write checkpoint '{target}': {exc}") from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    logger.debug("Saved checkpoint to %s (epoch %d).", target, epoch)
    return target


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = require_checkpoint(path)
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as exc:  # torch raises varied types (pickle/zip/runtime); re-raised with context
        raise CheckpointError(f"Could not read checkpoint '{checkpoint_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"Checkpoint '{checkpoint_path}' has unexpected structure.")
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise CheckpointError(f"Checkpoint '{checkpoint_path}' is missing keys: {missing}")
    if payload["version"] != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"Unsupported checkpoint version {payload['version']} (supported: {CHECKPOINT_VERSION})."
        )
    return payload


def restore_model(checkpoint: dict[str, Any], device: torch.device) -> MLP:
    model = MLP.from_init_config(checkpoint["model_init_config"])
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise CheckpointError(f"Checkpoint weights do not match the architecture: {exc}") from exc
    return model.to(device).eval()