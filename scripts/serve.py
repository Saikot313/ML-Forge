from __future__ import annotations

import argparse
import sys

from mlforge.api.app import create_app
from mlforge.config import InferenceConfig, load_config
from mlforge.logging_utils import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an MLForge model.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--config", default=None, help="Optional YAML; its inference section is used.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="Override inference device.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = parse_args(argv)
    configure_logging("INFO")
    inference_cfg: InferenceConfig = load_config(args.config).inference if args.config else InferenceConfig()
    if args.device:
        inference_cfg.device = args.device
    app = create_app(checkpoint_path=args.checkpoint, inference_config=inference_cfg)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())