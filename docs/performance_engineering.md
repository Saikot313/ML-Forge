# Performance Engineering

## Principles

1. Measure before claiming. This repository contains no benchmark numbers on purpose.
2. Compare like with like: identical weights, identical seeded inputs, identical batch sizes.
3. Separate model quality, numerical correctness, performance and resource use.
4. An optimisation is only an improvement on a machine where it was measured to be one.

## Measured versus estimated memory

| Quantity | Kind | Function |
|---|---|---|
| Process RSS (current) | measured | `process_memory_mb` |
| RSS start/end/peak over a region | measured (peak is sampled every 5 ms) | `MemoryTracker` |
| CUDA peak allocated | measured, CUDA only | `MemoryTracker`, `cuda_memory_mb` |
| Parameter count | exact | `count_parameters` |
| Parameter + buffer bytes | estimated | `estimate_model_memory_mb` |
| Forward activation memory | estimated (sum of Linear output widths × batch) | `estimate_activation_memory_mb` |

Estimates ignore allocator overhead, framework state and temporaries. The sampled RSS peak can miss
very short spikes. RSS includes everything in the process (data, libraries, other objects), so deltas
for small models are often dominated by noise.

## Optimisations implemented

| Technique | Where | When it helps | Caveat |
|---|---|---|---|
| Batch inference | `Predictor`, benchmark | Amortises per-call overhead; usually raises throughput | Raises latency per call and peak memory |
| `torch.inference_mode()` | `InferenceConfig.use_inference_mode` | Skips autograd bookkeeping | For tiny MLPs on CPU the gain may be negligible |
| Adaptive batch size | `adaptive_batch_size` | Recovers from OOM by halving the batch | Retries cost time; non-OOM errors are re-raised |
| Mixed precision (fp16 autocast) | `use_amp`, CUDA only | Tensor-core GPUs with larger models | Outputs differ slightly; tiny models may get slower; ignored on CPU |
| Pinned memory | `training.pin_memory` | Faster host-to-GPU copies | Only enabled when targeting CUDA |
| DataLoader workers | `training.num_workers` | CPU-heavy or I/O-heavy loading | For small in-memory tensors, workers are often slower |

Baseline and optimised variants are separately selectable: `baseline_autograd_fp32` and
`inference_mode_fp32` always run; `inference_mode_amp_fp16` runs only on CUDA.

## Benchmark methodology

- Warm-up iterations run first and are discarded.
- Each timed call is bracketed by `torch.cuda.synchronize` on CUDA (no-op on CPU).
- Timed region: model forward on device-resident inputs. Host-to-device transfer and preprocessing are
  excluded, which makes numbers comparable across variants but not an end-to-end latency.
- Statistics: mean, median, P95 (NumPy percentile, linear interpolation), min, max, std.
- Throughput = `batch_size / mean latency`.
- `speedup_vs_baseline_median` is the baseline median divided by the row's median; values above 1 mean
  lower latency than the baseline for the same device and batch size.
- `matches_reference` compares to a CPU fp32 reference under `rtol`/`atol`.
- CPU versus CUDA comparisons appear only when CUDA exists, and use the same weights and inputs.

## Interpreting results

- Compare rows from one run only. Close other applications, and keep thread settings fixed.
- With few iterations, P95 is dominated by the slowest sample; raise `benchmark.iterations`.
- Batch size 1 measures per-call overhead more than compute for small models.
- If two variants differ by less than the spread between min and max, treat them as equivalent.

## Profiling

`scripts/profile_inference.py` uses `torch.profiler` with shapes and memory recording and prints the
top operators by self CPU time; it writes a Chrome trace. Profiling overhead distorts timings: use the
profiler to find where time goes, and the benchmark to report latency.