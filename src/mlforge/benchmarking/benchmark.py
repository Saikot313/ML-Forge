from __future__ import annotations

import contextlib
import copy
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mlforge.config import BenchmarkConfig
from mlforge.data.loaders import make_dataloader
from mlforge.device import available_devices, describe_device, synchronize
from mlforge.exceptions import DataValidationError
from mlforge.logging_utils import get_logger
from mlforge.memory.profiler import MemoryTracker, count_parameters, estimate_model_memory_mb
from mlforge.models.mlp import MLP
from mlforge.numerics.stability import compare_arrays

logger = get_logger(__name__)
MS_PER_S = 1000.0
P95_PERCENTILE = 95.0
TABLE_COLUMNS = (
    "device",
    "variant",
    "batch_size",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "throughput_samples_per_s",
    "rss_peak_mb",
    "cuda_peak_allocated_mb",
    "matches_reference",
    "speedup_vs_baseline_median",
)


@dataclass(frozen=True)
class InferenceVariant:
    name: str
    use_inference_mode: bool
    use_amp: bool


BASELINE_VARIANT = InferenceVariant("baseline_autograd_fp32", False, False)
INFERENCE_MODE_VARIANT = InferenceVariant("inference_mode_fp32", True, False)
AMP_VARIANT = InferenceVariant("inference_mode_amp_fp16", True, True)


def variants_for_device(device: torch.device) -> list[InferenceVariant]:
    variants = [BASELINE_VARIANT, INFERENCE_MODE_VARIANT]
    if device.type == "cuda":
        variants.append(AMP_VARIANT)
    return variants


def make_forward(model: MLP, variant: InferenceVariant) -> Callable[[torch.Tensor], torch.Tensor]:
    def run(batch: torch.Tensor) -> torch.Tensor:
        grad_ctx = torch.inference_mode() if variant.use_inference_mode else contextlib.nullcontext()
        with grad_ctx:
            if variant.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return model(batch)
            return model(batch)

    return run


def latency_statistics(latencies_s: Sequence[float]) -> dict[str, float]:
    if len(latencies_s) == 0:
        raise DataValidationError("Cannot compute statistics from zero latency samples.")
    values = np.asarray(latencies_s, dtype=np.float64) * MS_PER_S
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, P95_PERCENTILE)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "std_ms": float(values.std()),
    }


def measure_latencies(
    forward: Callable[[torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    batch_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[float]:
    last_start = data.shape[0] - batch_size

    def batch_at(index: int) -> torch.Tensor:
        start = (index * batch_size) % (last_start + 1)
        return data[start : start + batch_size]

    for i in range(warmup):
        forward(batch_at(i))
    synchronize(device)

    latencies: list[float] = []
    for i in range(iterations):
        batch = batch_at(i)
        synchronize(device)
        started = time.perf_counter()
        forward(batch)
        synchronize(device)
        latencies.append(time.perf_counter() - started)
    return latencies


def _benchmark_one(
    forward: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    data: torch.Tensor,
    variant: InferenceVariant,
    batch_size: int,
    cfg: BenchmarkConfig,
    reference: np.ndarray,
) -> dict[str, Any]:
    output = forward(data[:batch_size]).detach().float().cpu().numpy()
    agreement = compare_arrays(output, reference, rtol=cfg.rtol, atol=cfg.atol)

    with MemoryTracker(device) as tracker:
        latencies = measure_latencies(
            forward, data, batch_size, device, cfg.warmup_iterations, cfg.iterations
        )
    stats = latency_statistics(latencies)
    memory = tracker.result
    row: dict[str, Any] = {
        "device": device.type,
        "variant": variant.name,
        "batch_size": batch_size,
        "mode": "single_sample" if batch_size == 1 else "batch",
        "warmup_iterations": cfg.warmup_iterations,
        "iterations": cfg.iterations,
        **stats,
        "throughput_samples_per_s": batch_size / (stats["mean_ms"] / MS_PER_S),
        "rss_peak_mb": memory.rss_peak_mb,
        "rss_delta_mb": memory.rss_delta_mb,
        "cuda_peak_allocated_mb": memory.cuda_peak_allocated_mb,
        "matches_reference": agreement.allclose,
        "max_abs_diff_vs_cpu_reference": agreement.max_abs_diff,
    }
    logger.info(
        "%s | %s | bs=%d | median=%.3f ms | p95=%.3f ms | %.0f samples/s",
        row["device"], variant.name, batch_size, row["median_ms"], row["p95_ms"],
        row["throughput_samples_per_s"],
    )
    return row


def _add_speedups(rows: list[dict[str, Any]]) -> None:
    baselines = {
        (r["device"], r["batch_size"]): r["median_ms"]
        for r in rows
        if r["variant"] == BASELINE_VARIANT.name
    }
    for row in rows:
        base = baselines.get((row["device"], row["batch_size"]))
        row["speedup_vs_baseline_median"] = (
            base / row["median_ms"] if base is not None and row["median_ms"] > 0 else None
        )


def _device_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(r["device"], r["variant"], r["batch_size"]): r for r in rows}
    comparison: list[dict[str, Any]] = []
    for (device, variant, batch_size), cpu_row in index.items():
        if device != "cpu":
            continue
        cuda_row = index.get(("cuda", variant, batch_size))
        if cuda_row is None:
            continue
        comparison.append(
            {
                "variant": variant,
                "batch_size": batch_size,
                "cpu_median_ms": cpu_row["median_ms"],
                "cuda_median_ms": cuda_row["median_ms"],
                "cpu_median_over_cuda_median": cpu_row["median_ms"] / cuda_row["median_ms"],
            }
        )
    return comparison


def run_benchmark(
    model: MLP,
    cfg: BenchmarkConfig,
    seed: int,
    devices: Sequence[str] | None = None,
) -> dict[str, Any]:
    device_list = available_devices(devices if devices is not None else cfg.devices)
    if not device_list:
        raise DataValidationError("No usable benchmark devices are available.")

    rng = np.random.default_rng(seed)
    data_cpu = torch.from_numpy(
        rng.standard_normal((cfg.num_samples, model.input_dim)).astype(np.float32)
    )

    reference_model = copy.deepcopy(model).cpu().eval()
    references: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for batch_size in cfg.batch_sizes:
            references[batch_size] = reference_model(data_cpu[:batch_size]).numpy()

    rows: list[dict[str, Any]] = []
    for device in device_list:
        device_model = copy.deepcopy(model).to(device).eval()
        data = data_cpu.to(device)
        for variant in variants_for_device(device):
            forward = make_forward(device_model, variant)
            for batch_size in cfg.batch_sizes:
                rows.append(
                    _benchmark_one(
                        forward, device, data, variant, batch_size, cfg, references[batch_size]
                    )
                )
        del device_model, data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _add_speedups(rows)
    return {
        "environment": {
            "devices": [describe_device(d) for d in device_list],
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_version": str(torch.__version__),
        },
        "model": {
            "input_dim": model.input_dim,
            "parameters": count_parameters(model),
            "memory_mb_estimated": estimate_model_memory_mb(model),
        },
        "settings": {
            "seed": seed,
            "warmup_iterations": cfg.warmup_iterations,
            "iterations": cfg.iterations,
            "batch_sizes": list(cfg.batch_sizes),
            "num_samples": cfg.num_samples,
            "rtol": cfg.rtol,
            "atol": cfg.atol,
        },
        "results": rows,
        "device_comparison": _device_comparison(rows),
        "notes": [
            "Timed region is model forward on device-resident inputs; transfer and preprocessing excluded.",
            "throughput_samples_per_s = batch_size / mean latency.",
            "speedup_vs_baseline_median > 1 means lower median latency than the baseline variant.",
            "rss_peak_mb is a sampled approximation of process RSS, not an exact allocator peak.",
            "cuda_peak_allocated_mb is null on CPU rows.",
            "matches_reference compares output to a CPU fp32 reference; fp16 autocast may legitimately differ.",
            "Results depend on hardware, thread count and background load; compare only within one run.",
        ],
    }


def format_results_table(results: dict[str, Any]) -> str:
    frame = pd.DataFrame(results["results"])
    columns = [c for c in TABLE_COLUMNS if c in frame.columns]
    return frame[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}")


def save_benchmark_results(
    results: dict[str, Any], output_dir: str | Path, stem: str = "benchmark"
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    csv_path = directory / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    pd.DataFrame(results["results"]).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def benchmark_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    worker_counts: Sequence[int],
    pin_memory: bool,
    seed: int,
    epochs: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workers in worker_counts:
        loader = make_dataloader(
            features, labels, batch_size, shuffle=False, seed=seed,
            num_workers=workers, pin_memory=pin_memory,
        )
        epoch_seconds: list[float] = []
        for _ in range(epochs):
            started = time.perf_counter()
            seen = 0
            for batch_features, _ in loader:
                seen += batch_features.shape[0]
            epoch_seconds.append(time.perf_counter() - started)
        median = float(np.median(epoch_seconds))
        rows.append(
            {
                "num_workers": workers,
                "pin_memory": pin_memory,
                "batch_size": batch_size,
                "epochs": epochs,
                "epoch_seconds": epoch_seconds,
                "median_epoch_s": median,
                "samples_per_s": seen / median if median > 0 else None,
            }
        )
        del loader
    return rows