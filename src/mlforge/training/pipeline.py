"""End-to-end training workflow producing a structured results file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlforge.checkpointing import CheckpointContext, build_metadata
from mlforge.config import ExperimentConfig
from mlforge.data.loaders import make_dataloader
from mlforge.data.pipeline import prepare_data
from mlforge.device import describe_device, resolve_device, resolve_pin_memory
from mlforge.evaluation.comparison import compare_models, prediction_agreement
from mlforge.evaluation.metrics import compute_classification_metrics
from mlforge.inference.predictor import Predictor
from mlforge.logging_utils import get_logger
from mlforge.memory.profiler import (
    MemoryTracker,
    count_parameters,
    estimate_model_memory_mb,
)
from mlforge.models.baseline import LogisticRegressionBaseline
from mlforge.models.mlp import build_mlp
from mlforge.numerics.stability import compare_arrays, stable_softmax
from mlforge.seed import set_global_seed
from mlforge.training.reporting import plot_training_history
from mlforge.training.trainer import Trainer

logger = get_logger(__name__)
CONSISTENCY_RTOL = 1e-5
CONSISTENCY_ATOL = 1e-6


def run_training(cfg: ExperimentConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Train the MLP and baseline, evaluate both on the test split, and save results.

    Outputs under ``<output_dir>/<experiment_name>/``: ``checkpoints/best_model.pt``,
    ``results.json`` and ``training_history.png``.
    """
    run_dir = Path(output_dir or cfg.output_dir) / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"

    set_global_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    prepared = prepare_data(cfg.data, cfg.seed)
    splits = prepared.splits

    pin = resolve_pin_memory(cfg.training.pin_memory, device)
    train_loader = make_dataloader(
        splits.X_train, splits.y_train, cfg.training.batch_size, True, cfg.seed,
        cfg.training.num_workers, pin,
    )
    val_loader = make_dataloader(
        splits.X_val, splits.y_val, cfg.inference.batch_size, False, cfg.seed,
        cfg.training.num_workers, pin,
    )

    model = build_mlp(cfg.model, prepared.n_features, prepared.n_classes).to(device)
    context = CheckpointContext(
        config=cfg.to_dict(),
        preprocessor_state=prepared.preprocessor.to_dict(),
        feature_names=prepared.feature_names,
        seed=cfg.seed,
    )
    trainer = Trainer(model, cfg.training, cfg.diagnostics, device, checkpoint_path, context)

    with MemoryTracker(device) as tracker:
        history = trainer.fit(train_loader, val_loader)
    history_dict = history.to_dict()

    # --- Predictive quality (identical metric code for both models) -------
    predictor = Predictor.from_checkpoint(checkpoint_path, cfg.inference, device=cfg.device)
    mlp_proba = predictor.predict_proba_preprocessed(splits.X_test)
    baseline = LogisticRegressionBaseline(cfg.baseline, cfg.seed).fit(splits.X_train, splits.y_train)
    baseline_proba = baseline.predict_proba(splits.X_test)
    metrics = {
        "mlp": compute_classification_metrics(splits.y_test, mlp_proba),
        "logistic_regression": compute_classification_metrics(splits.y_test, baseline_proba),
    }
    comparison = compare_models(metrics, reference="logistic_regression", candidate="mlp")

    # --- Numerical correctness: in-memory model vs reloaded checkpoint ----
    model.eval()
    with torch.inference_mode():
        in_memory_logits = model(torch.from_numpy(splits.X_test).to(device)).float().cpu().numpy()
    in_memory_proba = stable_softmax(in_memory_logits.astype(np.float64), axis=1)
    roundtrip = compare_arrays(
        in_memory_proba, mlp_proba, rtol=CONSISTENCY_RTOL, atol=CONSISTENCY_ATOL
    )

    results: dict[str, Any] = {
        "experiment": {
            "name": cfg.experiment_name,
            "metadata": build_metadata(cfg.seed, device),
            "device_info": describe_device(device),
            "deterministic_requested": cfg.deterministic,
            "reproducibility_note": (
                "Repeatable on the same hardware/library versions; not guaranteed across "
                "different hardware, library versions or non-deterministic GPU kernels."
            ),
            "config": cfg.to_dict(),
        },
        "data": {
            "n_rows": prepared.report.n_rows,
            "n_features": prepared.report.n_features,
            "n_classes": prepared.report.n_classes,
            "class_counts": prepared.report.class_counts,
            "missing_fraction_raw": prepared.report.missing_fraction,
            "split_sizes": {
                "train": int(len(splits.y_train)),
                "val": int(len(splits.y_val)),
                "test": int(len(splits.y_test)),
            },
        },
        "predictive_quality": {"test_metrics": metrics, "comparison": comparison},
        "numerical_correctness": {
            "checkpoint_roundtrip_vs_in_memory": roundtrip.to_dict(),
            "mlp_vs_baseline_agreement": prediction_agreement(mlp_proba, baseline_proba),
        },
        "inference_performance": "Not measured here. Run scripts/benchmark.py for latency/throughput.",
        "resource_consumption": {
            "training_measured": tracker.result.to_dict(),
            "model_parameters": count_parameters(model),
            "model_memory_mb_estimated": estimate_model_memory_mb(model),
        },
        "training_history": history_dict,
        "artifacts": {"checkpoint": str(checkpoint_path)},
    }

    results_path = run_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    plot_training_history(history_dict, run_dir / "training_history.png")
    logger.info("Training complete. Results written to %s", results_path)
    return results