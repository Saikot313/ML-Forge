from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch
import torch.nn as nn

BYTES_PER_MB = 1024**2
DEFAULT_SAMPLE_INTERVAL_S = 0.005


def process_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / BYTES_PER_MB


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def estimate_model_memory_mb(model: nn.Module, include_buffers: bool = True) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    if include_buffers:
        total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / BYTES_PER_MB


def estimate_activation_memory_mb(model: nn.Module, batch_size: int, bytes_per_element: int = 4) -> float:
    width = sum(m.out_features for m in model.modules() if isinstance(m, nn.Linear))
    return batch_size * width * bytes_per_element / BYTES_PER_MB


def cuda_memory_mb(device: torch.device | None = None) -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / BYTES_PER_MB,
        "reserved_mb": torch.cuda.memory_reserved(device) / BYTES_PER_MB,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / BYTES_PER_MB,
    }


@dataclass(frozen=True)
class MemoryMeasurement:
    rss_start_mb: float
    rss_end_mb: float
    rss_peak_mb: float
    rss_delta_mb: float
    cuda_peak_allocated_mb: float | None
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryTracker:
    def __init__(
        self,
        device: torch.device | None = None,
        sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    ) -> None:
        self._device = device if device is not None and device.type == "cuda" else None
        self._interval = sample_interval_s
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0.0
        self._start_rss = 0.0
        self._start_time = 0.0
        self._result: MemoryMeasurement | None = None

    def _rss_mb(self) -> float:
        return self._process.memory_info().rss / BYTES_PER_MB

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._peak = max(self._peak, self._rss_mb())

    def __enter__(self) -> MemoryTracker:
        if self._device is not None:
            torch.cuda.synchronize(self._device)
            torch.cuda.reset_peak_memory_stats(self._device)
        self._start_rss = self._rss_mb()
        self._peak = self._start_rss
        self._stop.clear()
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._device is not None:
            torch.cuda.synchronize(self._device)
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        end_rss = self._rss_mb()
        self._peak = max(self._peak, end_rss)
        cuda_peak = (
            torch.cuda.max_memory_allocated(self._device) / BYTES_PER_MB
            if self._device is not None
            else None
        )
        self._result = MemoryMeasurement(
            rss_start_mb=self._start_rss,
            rss_end_mb=end_rss,
            rss_peak_mb=self._peak,
            rss_delta_mb=end_rss - self._start_rss,
            cuda_peak_allocated_mb=cuda_peak,
            duration_s=time.perf_counter() - self._start_time,
        )
        return False

    @property
    def result(self) -> MemoryMeasurement:
        if self._result is None:
            raise RuntimeError("MemoryTracker.result is available only after the context exits.")
        return self._result