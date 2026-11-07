from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlforge.benchmarking.benchmark import (
    benchmark_dataloader,
    format_results_table,
    run_benchmark,
    save_benchmark_results,
)
from mlforge.checkpointing import load_checkpoint, restore_model
from mlforge.config import load_config
from mlforge.data.pipeline import prepare_data
from mlforge.device import resolve_device, resolve_pin_memory
from mlforge.exceptions import MLForgeError
from mlforge.logging_utils import configure_logging, get_logger
from mlforge.models.mlp import build_mlp
from mlforge.seed import set_global_seed

logger = get_logger("scripts.benchmark")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MLForge inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Use trained weights (optional).")
    parser.add_argument("--output-dir", default=None, help="Default: <output_dir>/benchmarks.")
    parser.add_argument("--devices", nargs="+", default=None, help="e.g. cpu cuda (cuda skipped if absent).")
    parser.add_argument("--dataloader", action="store_true", help="Also benchmark DataLoader workers.")
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2], help="Worker counts to test.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging("INFO")
    try:
        cfg = load_config(args.config)
        set_global_seed(cfg.seed, cfg.deterministic)
        if args.checkpoint:
            model = restore_model(load_checkpoint(args.checkpoint), resolve_device("cpu"))
            weights_source = f"checkpoint:{args.checkpoint}"
        else:
            model = build_mlp(cfg.model, cfg.data.n_features, cfg.data.n_classes).eval()
            weights_source = "random_init (latency does not depend on weight values)"

        results = run_benchmark(model, cfg.benchmark, cfg.seed, args.devices)
        results["model"]["weights_source"] = weights_source
        output_dir = Path(args.output_dir or Path(cfg.output_dir) / "benchmarks")
        paths = save_benchmark_results(results, output_dir)

        print(format_results_table(results))
        for entry in results["device_comparison"]:
            print(
                f"CPU/CUDA median ratio | {entry['variant']} bs={entry['batch_size']}: "
                f"{entry['cpu_median_over_cuda_median']:.2f}"
            )
        print(f"\nWrote {paths['json']} and {paths['csv']}")

        if args.dataloader:
            prepared = prepare_data(cfg.data, cfg.seed)
            pin = resolve_pin_memory(cfg.training.pin_memory, resolve_device(cfg.device))
            rows = benchmark_dataloader(
                prepared.splits.X_train, prepared.splits.y_train, cfg.training.batch_size,
                args.workers, pin, cfg.seed,
            )
            loader_path = output_dir / "dataloader_benchmark.json"
            with loader_path.open("w", encoding="utf-8") as handle:
                json.dump(rows, handle, indent=2)
            for row in rows:
                print(f"num_workers={row['num_workers']}: {row['samples_per_s']:.0f} samples/s (median epoch)")
            print(f"Wrote {loader_path}")
    except MLForgeError as exc:
        logger.error("Benchmark failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__": 
    sys.exit(main())