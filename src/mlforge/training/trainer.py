"""Training loop with diagnostics, early stopping and best-model checkpointing."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from mlforge.checkpointing import CheckpointContext, save_checkpoint
from mlforge.config import DiagnosticsConfig, TrainingConfig
from mlforge.diagnostics.checks import check_loss_finite, inspect_gradients
from mlforge.exceptions import NumericalInstabilityError
from mlforge.logging_utils import get_logger
from mlforge.models.mlp import MLP
from mlforge.numerics.gradients import clip_gradients, gradient_norm
from mlforge.training.early_stopping import EarlyStopping

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    """Per-epoch training record plus summary fields."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    gradient_warning_epochs: list[int] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_completed: int = 0
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trainer:
    """Trains an :class:`MLP` with cross-entropy loss."""

    def __init__(
        self,
        model: MLP,
        config: TrainingConfig,
        diagnostics: DiagnosticsConfig,
        device: torch.device,
        checkpoint_path: Path | None = None,
        checkpoint_context: CheckpointContext | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.diagnostics = diagnostics
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_context = checkpoint_context
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = self.model.parameters()
        cfg = self.config
        if cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        return torch.optim.SGD(
            params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
        if self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
            )
        return None

    def _train_one_epoch(self, loader: DataLoader, epoch: int, history: TrainingHistory) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        grad_norms: list[float] = []
        for step, (features, targets) in enumerate(loader):
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(features)
            loss = MLP.compute_loss(logits, targets)
            loss_value = check_loss_finite(loss, f"epoch {epoch}, step {step}")
            loss.backward()

            if step == 0:  # inspect raw (unclipped) gradients once per epoch
                report = inspect_gradients(
                    self.model,
                    self.diagnostics.gradient_explosion_threshold,
                    self.diagnostics.gradient_vanishing_threshold,
                )
                if report.has_issues:
                    history.gradient_warning_epochs.append(epoch)
                    logger.warning("Gradient diagnostics at epoch %d: %s", epoch, report.summary())

            if self.config.max_grad_norm is not None:
                norm = clip_gradients(self.model.parameters(), self.config.max_grad_norm)
            else:
                norm = gradient_norm(self.model.parameters())
                if norm != norm or norm == float("inf"):
                    raise NumericalInstabilityError(
                        f"Non-finite gradient norm at epoch {epoch}, step {step}."
                    )
            self.optimizer.step()

            batch_size = targets.shape[0]
            total_loss += loss_value * batch_size
            total_samples += batch_size
            grad_norms.append(norm)
        return total_loss / total_samples, sum(grad_norms) / len(grad_norms)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Loss, accuracy and macro-F1 on a loader."""
        self.model.eval()
        total_loss = 0.0
        total = 0
        predictions: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for features, targets in loader:
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(features)
            total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
            total += targets.shape[0]
            predictions.append(logits.argmax(dim=1).cpu())
            labels.append(targets.cpu())
        loss = check_loss_finite(total_loss / total, "validation")
        y_pred = torch.cat(predictions).numpy()
        y_true = torch.cat(labels).numpy()
        return {
            "loss": loss,
            "accuracy": float((y_pred == y_true).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        """Train, select the best epoch by validation loss, and restore its weights."""
        history = TrainingHistory()
        stopper = EarlyStopping(
            self.config.early_stopping_patience, self.config.early_stopping_min_delta
        )
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(1, self.config.epochs + 1):
            started = time.perf_counter()
            train_loss, mean_grad_norm = self._train_one_epoch(train_loader, epoch, history)
            val = self.evaluate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step(val["loss"])

            history.train_loss.append(train_loss)
            history.val_loss.append(val["loss"])
            history.val_accuracy.append(val["accuracy"])
            history.val_macro_f1.append(val["macro_f1"])
            history.learning_rate.append(self.optimizer.param_groups[0]["lr"])
            history.grad_norm.append(mean_grad_norm)
            history.epoch_seconds.append(time.perf_counter() - started)
            history.epochs_completed = epoch

            if stopper.step(val["loss"]):
                history.best_epoch = epoch
                history.best_val_loss = val["loss"]
                best_state = copy.deepcopy(
                    {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                )
                self._save_best(epoch, val, history)

            logger.info(
                "epoch %03d | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f grad_norm=%.3f",
                epoch, train_loss, val["loss"], val["accuracy"], val["macro_f1"], mean_grad_norm,
            )
            if stopper.should_stop:
                history.stopped_early = True
                logger.info("Early stopping at epoch %d (best epoch %d).", epoch, history.best_epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return history

    def _save_best(self, epoch: int, val: dict[str, float], history: TrainingHistory) -> None:
        if self.checkpoint_path is None or self.checkpoint_context is None:
            return
        save_checkpoint(
            self.checkpoint_path,
            self.model,
            self.optimizer,
            epoch,
            val["loss"],
            history.to_dict(),
            self.checkpoint_context,
            metrics={"validation": val},
        )