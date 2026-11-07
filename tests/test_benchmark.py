import json

import numpy as np
import pandas as pd
import pytest
import torch

from mlforge.benchmarking import (
    benchmark_dataloader,
    format_results_table,
    latency_statistics,
    profile_inference,
    run_benchmark,
    save_benchmark_results,
)
from mlforge.config import BenchmarkConfig
from mlforge.exceptions import ConfigError, DataValidationError
from mlforge.models.mlp import MLP

CFG = BenchmarkConfig(warmup_iterations=1, iterations=3, batch_sizes=(1, 8), num_samples=32,
                      devices=("cpu", "cuda"))
REQUIRED_ROW_KEYS = {
    "device", "variant", "batch_size", "mode", "mean_ms", "median_ms", "p95_ms", "min_ms",
    "max_ms", "throughput_samples_per_s", "rss_peak_mb", "cuda_peak_allocated_mb",
    "matches_reference", "speedup_vs_baseline_median",
}


@pytest.fixture(scope="module")
def results() -> dict:
    model = MLP(8, [8, 4], 2).eval()
    return run_benchmark(model, CFG, seed=0)


def test_latency_statistics_known_values() -> None:
    stats = latency_statistics([i / 1000 for i in range(1, 101)])
    assert stats["median_ms"] == pytest.approx(50.5)
    assert stats["p95_ms"] == pytest.approx(95.05)
    assert stats["min_ms"] == pytest.approx(1.0) and stats["max_ms"] == pytest.approx(100.0)
    with pytest.raises(DataValidationError):
        latency_statistics([])


def test_result_structure(results: dict) -> None:
    assert {"environment", "model", "settings", "results", "device_comparison", "notes"} <= set(results)
    for row in results["results"]:
        assert REQUIRED_ROW_KEYS <= set(row)


def test_rows_cover_variants_and_batch_sizes(results: dict) -> None:
    cpu_rows = [r for r in results["results"] if r["device"] == "cpu"]
    assert {r["variant"] for r in cpu_rows} == {"baseline_autograd_fp32", "inference_mode_fp32"}
    assert {r["batch_size"] for r in cpu_rows} == {1, 8}
    assert {r["mode"] for r in cpu_rows} == {"single_sample", "batch"}


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks CUDA is skipped when absent")
def test_cuda_skipped_without_gpu(results: dict) -> None:
    assert {r["device"] for r in results["results"]} == {"cpu"}
    assert results["device_comparison"] == []


def test_statistics_are_internally_consistent(results: dict) -> None:
    for row in results["results"]:
        assert 0 < row["min_ms"] <= row["median_ms"] <= row["max_ms"]
        assert row["median_ms"] <= row["p95_ms"] <= row["max_ms"] + 1e-9
        assert row["throughput_samples_per_s"] == pytest.approx(
            row["batch_size"] / (row["mean_ms"] / 1000.0)
        )
        assert row["rss_peak_mb"] > 0


def test_fp32_cpu_variants_match_reference_and_baseline_speedup_is_one(results: dict) -> None:
    for row in results["results"]:
        if row["device"] == "cpu":
            assert row["matches_reference"] is True
        if row["variant"] == "baseline_autograd_fp32":
            assert row["speedup_vs_baseline_median"] == pytest.approx(1.0)


def test_save_and_format(results: dict, tmp_path) -> None:
    paths = save_benchmark_results(results, tmp_path / "bench")
    with paths["json"].open(encoding="utf-8") as handle:
        assert len(json.load(handle)["results"]) == len(results["results"])
    assert len(pd.read_csv(paths["csv"])) == len(results["results"])
    assert "median_ms" in format_results_table(results)


def test_benchmark_config_validation() -> None:
    with pytest.raises(ConfigError):
        BenchmarkConfig(batch_sizes=(64,), num_samples=8)
    with pytest.raises(ConfigError):
        BenchmarkConfig(iterations=0)


def test_dataloader_benchmark() -> None:
    x = np.random.default_rng(0).normal(size=(256, 4)).astype(np.float32)
    y = np.zeros(256, dtype=np.int64)
    rows = benchmark_dataloader(x, y, 32, [0], pin_memory=False, seed=0, epochs=2)
    assert rows[0]["num_workers"] == 0 and rows[0]["samples_per_s"] > 0
    assert len(rows[0]["epoch_seconds"]) == 2


def test_profile_inference_reports_operators(tmp_path) -> None:
    trace = tmp_path / "trace.json"
    result = profile_inference(MLP(8, [8], 2), 4, torch.device("cpu"), steps=2, warmup=1,
                               trace_path=trace)
    assert result["top_operators"] and "table" in result
    assert trace.is_file()