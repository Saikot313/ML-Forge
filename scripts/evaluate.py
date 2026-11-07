from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlforge.checkpointing import load_checkpoint
from mlforge.config import ExperimentConfig
from mlforge.data.pipeline import prepare_data
from mlforge.evaluation.comparison import compare_models, prediction_agreement
from mlforge.evaluation.metrics import compute_classification_metrics
from mlforge.exceptions import DataValidationError, MLForgeError
from mlforge.inference.predictor import Predictor
from mlforge.logging_utils import configure_logging, get_logger
from mlforge.models.baseline import LogisticRegressionBaseline
from mlforge.seed import set_global_seed

logger = get_logger("scripts.evaluate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MLForge checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--device", default=None, help="Inference device (auto|cpu|cuda).")
    parser.add_argument("--output", default=None, help="Where to write evaluation JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging("INFO")
    try:
        checkpoint = load_checkpoint(args.checkpoint)
        cfg = ExperimentConfig.from_dict(checkpoint["config"])
        set_global_seed(cfg.seed, cfg.deterministic)
        prepared = prepare_data(cfg.data, cfg.seed)
        predictor = Predictor.from_checkpoint(args.checkpoint, cfg.inference, device=args.device)
        if prepared.feature_names != predictor.feature_names:
            raise DataValidationError("Regenerated data features differ from the checkpoint's features.")

        splits = prepared.splits
        mlp_proba = predictor.predict_proba_preprocessed(splits.X_test)
        baseline = LogisticRegressionBaseline(cfg.baseline, cfg.seed).fit(splits.X_train, splits.y_train)
        baseline_proba = baseline.predict_proba(splits.X_test)

        metrics = {
            "mlp": compute_classification_metrics(splits.y_test, mlp_proba),
            "logistic_regression": compute_classification_metrics(splits.y_test, baseline_proba),
        }
        report = {
            "checkpoint": str(args.checkpoint),
            "test_metrics": metrics,
            "comparison": compare_models(metrics, "logistic_regression", "mlp"),
            "mlp_vs_baseline_agreement": prediction_agreement(mlp_proba, baseline_proba),
            "note": "Baseline is refit from the stored config; MLP weights come from the checkpoint.",
        }
    except MLForgeError as exc:
        logger.error("Evaluation failed: %s", exc)
        return 1

    output = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    for name, values in metrics.items():
        print(f"{name:<20} acc={values['accuracy']:.4f} f1={values['f1']:.4f} log_loss={values['log_loss']:.4f}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())