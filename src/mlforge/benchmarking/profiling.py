"""PyTorch Profiler wrapper for inference."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from mlforge.device import synchronize
from mlforge.models.mlp import MLP


def profile_inference(
    model: MLP,
    batch_size: int,
    device: torch.device,
    steps: int = 10,
    warmup: int = 3,
    seed: int = 0,
    trace_path: str | Path | None = None,
    top_k: int = 15,
    use_inference_mode: bool = True,
) -> dict[str, Any]:
    model = model.to(device).eval()
    generator = torch.Generator().manual_seed(seed)
    batch = torch.randn(batch_size, model.input_dim, generator=generator).to(device)

    def grad_context() -> Any:
        return torch.inference_mode() if use_inference_mode else contextlib.nullcontext()

    with grad_context():
        for _ in range(warmup):
            model(batch)
    synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with grad_context():
            for _ in range(steps):
                model(batch)
        synchronize(device)

    events = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    top_operators = [
        {
            "name": event.key,
            "count": int(event.count),
            "self_cpu_time_total_us": float(event.self_cpu_time_total),
            "cpu_time_total_us": float(event.cpu_time_total),
            "self_cpu_memory_bytes": int(event.self_cpu_memory_usage),
        }
        for event in events[:top_k]
    ]
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=top_k)

    written_trace: str | None = None
    if trace_path is not None:
        trace = Path(trace_path)
        trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace))
        written_trace = str(trace)

    return {
        "device": device.type,
        "batch_size": batch_size,
        "steps": steps,
        "use_inference_mode": use_inference_mode,
        "top_operators": top_operators,
        "table": table,
        "trace_path": written_trace,
    }