from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlforge.benchmarking.profiling import profile_inference
from mlforge.checkpointing import load_checkpoint, restore_model
from mlforge.config import load_config
from mlforge.device import resolve_device
from mlforge.exceptions import MLForgeError
from mlforge.logging_utils import configure_logging, get_logger
from mlforge.models.mlp import build_mlp
from mlforge.seed import set_global_seed

logger = get_logger("scripts.profile_inference")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile MLForge inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default=None, help="Default: inference.device from config.")
    parser.add_argument("--no-inference-mode", action="store_true", help="Profile the autograd baseline.")
    parser.add_argument("--output-dir", default=None, help="Default: <output_dir>/profiles.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging("INFO")
    try:
        cfg = load_config(args.config)
        set_global_seed(cfg.seed, cfg.deterministic)
        device = resolve_device(args.device or cfg.inference.device)
        if args.checkpoint:
            model = restore_model(load_checkpoint(args.checkpoint), device)
        else:
            model = build_mlp(cfg.model, cfg.data.n_features, cfg.data.n_classes)

        output_dir = Path(args.output_dir or Path(cfg.output_dir) / "profiles")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = profile_inference(
            model, args.batch_size, device, steps=args.steps, seed=cfg.seed,
            trace_path=output_dir / "inference_trace.json",
            use_inference_mode=not args.no_inference_mode,
        )
    except MLForgeError as exc:
        logger.error("Profiling failed: %s", exc)
        return 1

    (output_dir / "inference_profile.txt").write_text(result["table"], encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "table"}
    with (output_dir / "inference_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(result["table"])
    print(f"\nWrote profile files to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())