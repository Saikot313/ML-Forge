import numpy as np
import pytest
import torch

from mlforge.memory import (
    MemoryTracker,
    count_parameters,
    cuda_memory_mb,
    estimate_activation_memory_mb,
    estimate_model_memory_mb,
    process_memory_mb,
)
from mlforge.models.mlp import MLP

BYTES_PER_MB = 1024**2
ALLOCATION_MB = 64
MIN_EXPECTED_GROWTH_MB = 20


def test_process_memory_is_positive() -> None:
    assert process_memory_mb() > 0


def test_parameter_count_and_memory_estimate() -> None:
    model = MLP(4, [8], 2)
    params = (4 * 8 + 8) + (8 * 2 + 2)
    assert count_parameters(model) == params
    assert estimate_model_memory_mb(model) == pytest.approx(params * 4 / BYTES_PER_MB)


def test_activation_estimate() -> None:
    model = MLP(4, [8], 2)
    assert estimate_activation_memory_mb(model, batch_size=100) == pytest.approx(
        100 * (8 + 2) * 4 / BYTES_PER_MB
    )


def test_memory_tracker_observes_allocation() -> None:
    with MemoryTracker() as tracker:
        block = np.ones(ALLOCATION_MB * BYTES_PER_MB // 8)  # written, so pages are resident
    result = tracker.result
    assert block.size > 0
    assert result.rss_peak_mb >= result.rss_start_mb
    assert result.rss_end_mb - result.rss_start_mb > MIN_EXPECTED_GROWTH_MB
    assert result.duration_s > 0 and result.cuda_peak_allocated_mb is None


def test_tracker_result_unavailable_inside_context() -> None:
    with MemoryTracker() as tracker:
        with pytest.raises(RuntimeError):
            _ = tracker.result


def test_tracker_does_not_suppress_exceptions() -> None:
    with pytest.raises(ValueError):
        with MemoryTracker():
            raise ValueError("boom")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU-only fallback")
def test_cuda_memory_is_none_without_cuda() -> None:
    assert cuda_memory_mb() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tracker_reports_cuda_peak() -> None:
    device = torch.device("cuda")
    with MemoryTracker(device) as tracker:
        _ = torch.ones(1024, 1024, device=device)
    assert tracker.result.cuda_peak_allocated_mb is not None
    assert tracker.result.cuda_peak_allocated_mb > 0