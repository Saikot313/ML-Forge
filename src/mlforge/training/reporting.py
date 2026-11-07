"""Training report plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_training_history(history: dict[str, Any], output_path: Path) -> Path:
    """Plot loss and validation accuracy per epoch to ``output_path`` (PNG)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    if history.get("best_epoch"):
        ax_loss.axvline(history["best_epoch"], linestyle="--", color="gray", label="best epoch")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.legend()
    ax_loss.set_title("Loss")

    ax_acc.plot(epochs, history["val_accuracy"], label="val accuracy")
    ax_acc.plot(epochs, history["val_macro_f1"], label="val macro-F1")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()
    ax_acc.set_title("Validation metrics")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path