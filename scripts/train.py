from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlforge.config import load_config
from mlforge.exceptions import MLForgeError
from mlforge.logging_utils import configure_logging, get_logger
from mlforge.training.pipeline import run_training

logger = get_logger("scripts.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLForge models.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--output-dir", default=None, help="Override config output_dir.")
    parser.add_argument("--experiment-name", default=None, help="Override experiment_name.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--device", default=None, help="Override device (auto|cpu|cuda).")
    return parser.parse_args(argv)


def print_summary(results: dict) -> None:
    metrics = results["predictive_quality"]["test_metrics"]
    print("\nTest-set metrics (single split, single seed):")
    for model_name, values in metrics.items():
        auc = values["roc_auc"]
        auc_text = "n/a" if auc is None else f"{auc:.4f}"
        print(
            f"  {model_name:<20} acc={values['accuracy']:.4f} f1={values['f1']:.4f} "
            f"roc_auc={auc_text} log_loss={values['log_loss']:.4f}"
        )
    history = results["training_history"]
    print(
        f"\nBest epoch: {history['best_epoch']} / {history['epochs_completed']} "
        f"(stopped early: {history['stopped_early']})"
    )
    print(f"Checkpoint: {results['artifacts']['checkpoint']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging("INFO")
    try:
        cfg = load_config(args.config)
        if args.seed is not None:
            cfg.seed = args.seed
        if args.device:
            cfg.device = args.device
        if args.experiment_name:
            cfg.experiment_name = args.experiment_name
        output_dir = Path(args.output_dir or cfg.output_dir)
        configure_logging(cfg.log_level, output_dir / cfg.experiment_name / "train.log")
        results = run_training(cfg, output_dir)
    except MLForgeError as exc:
        logger.error("Training failed: %s", exc)
        return 1
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())