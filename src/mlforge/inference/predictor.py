from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlforge.checkpointing import load_checkpoint, restore_model
from mlforge.config import InferenceConfig
from mlforge.data.preprocessing import TabularPreprocessor
from mlforge.data.validation import validate_feature_matrix
from mlforge.device import resolve_device
from mlforge.diagnostics.checks import check_model_outputs, is_oom_error
from mlforge.logging_utils import get_logger
from mlforge.models.mlp import MLP
from mlforge.numerics.stability import stable_softmax

logger = get_logger(__name__)


class Predictor:
    def __init__(
        self,
        model: MLP,
        preprocessor: TabularPreprocessor,
        device: torch.device,
        config: InferenceConfig | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.config = config or InferenceConfig()
        self.device = device
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.feature_names = feature_names or []
        self.n_features = model.input_dim
        self.n_classes = model.output_dim

        self.use_amp = self.config.use_amp
        if self.use_amp and device.type != "cuda":
            logger.warning("use_amp requested but device is %s; AMP disabled (CUDA only).", device.type)
            self.use_amp = False

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        config: InferenceConfig | None = None,
        device: str | torch.device | None = None,
    ) -> Predictor:
        cfg = config or InferenceConfig()
        resolved = resolve_device(device if device is not None else cfg.device)
        checkpoint = load_checkpoint(path)
        model = restore_model(checkpoint, resolved)
        preprocessor = TabularPreprocessor.from_dict(checkpoint["preprocessor"])
        logger.info("Loaded checkpoint %s onto %s.", path, resolved)
        return cls(model, preprocessor, resolved, cfg, checkpoint["feature_names"])

    def _grad_context(self) -> Any:
        if self.config.use_inference_mode:
            return torch.inference_mode()
        return contextlib.nullcontext()

    def _forward_all(self, features: np.ndarray, batch_size: int) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        outputs: list[torch.Tensor] = []
        with self._grad_context():
            for start in range(0, tensor.shape[0], batch_size):
                batch = tensor[start : start + batch_size].to(self.device, non_blocking=True)
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(batch)
                else:
                    logits = self.model(batch)
                outputs.append(logits.detach().float().cpu())
        return torch.cat(outputs).numpy()

    def predict_logits_preprocessed(
        self, features: np.ndarray, batch_size: int | None = None
    ) -> np.ndarray:
        if features.shape[0] == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        size = batch_size or self.config.batch_size
        while True:
            try:
                logits = self._forward_all(features, size)
                break
            except (RuntimeError, MemoryError) as exc:
                can_shrink = size > self.config.min_batch_size
                if not (self.config.adaptive_batch_size and is_oom_error(exc) and can_shrink):
                    raise
                new_size = max(self.config.min_batch_size, size // 2)
                logger.warning("OOM at batch_size=%d; retrying with %d.", size, new_size)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                size = new_size
        check_model_outputs(torch.from_numpy(logits), self.n_classes)
        return logits

    def predict_proba_preprocessed(
        self, features: np.ndarray, batch_size: int | None = None
    ) -> np.ndarray:
        logits = self.predict_logits_preprocessed(features, batch_size)
        return stable_softmax(logits.astype(np.float64), axis=1)

    def _prepare(self, raw_features: Any) -> np.ndarray:
        validated = validate_feature_matrix(raw_features, self.n_features, name="features")
        return self.preprocessor.transform(validated)

    def predict_proba(self, raw_features: Any, batch_size: int | None = None) -> np.ndarray:
        return self.predict_proba_preprocessed(self._prepare(raw_features), batch_size)

    def predict(self, raw_features: Any, batch_size: int | None = None) -> np.ndarray:
        return self.predict_proba(raw_features, batch_size).argmax(axis=1)

    def predict_one(self, features: Sequence[float | None]) -> dict[str, Any]:
        proba = self.predict_proba([list(features)], batch_size=1)[0]
        return {"label": int(proba.argmax()), "probabilities": [float(p) for p in proba]}