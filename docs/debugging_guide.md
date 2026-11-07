# Debugging Guide

Each diagnostic raises a specific exception with likely causes, or reports without raising.

| Symptom | Detector | Behaviour | Typical causes |
|---|---|---|---|
| NaN loss | `check_loss_finite` | `NumericalInstabilityError` naming epoch/step | Learning rate too high, NaN/Inf inputs, log(0) |
| Infinite loss | `check_loss_finite` | `NumericalInstabilityError` | Overflow, extreme feature scale, divergence |
| Invalid model output | `check_model_outputs` | `DataValidationError` for shape, `NumericalInstabilityError` for NaN/Inf | Bad weights, fp16 overflow, wrong class count |
| Exploding gradients | `inspect_gradients` | Report lists layers above `gradient_explosion_threshold`; trainer logs a warning | LR too high, no normalisation, deep stack |
| Vanishing/zero gradients | `inspect_gradients` | Lists layers below `gradient_vanishing_threshold` | Dead ReLUs, saturated tanh, wrong init |
| Parameters without gradients | `inspect_gradients` | `layers_without_grad` | Frozen layers, detached graph, called before `backward()` |
| Non-finite gradients | `clip_gradients` / trainer | `NumericalInstabilityError` | NaN propagation |
| Wrong tensor rank/width | `check_tensor`, `MLP.forward` | `DataValidationError` with shapes | Wrong feature count or missing batch dim |
| Wrong dtype | `check_tensor(dtype=...)` | `DataValidationError` | float64 vs float32, integer inputs |
| Device mismatch | `check_tensor(device=...)`, `check_model_device` | `DeviceError` | Model on GPU, data on CPU |
| CUDA requested, unavailable | `resolve_device` | `DeviceError` | CPU-only install or no driver |
| Missing/corrupt checkpoint | `require_checkpoint`, `load_checkpoint` | `CheckpointError` | Wrong path, run training first, truncated file |
| Out of memory | `is_oom_error`, `Predictor` | Halves batch size and retries when `adaptive_batch_size` is on | Batch too large for device |

Notes:

- The trainer inspects gradients on the first step of every epoch **before** clipping, so clipping
  cannot hide an explosion. Epochs with issues are recorded in `history.gradient_warning_epochs`.
- Errors are never swallowed. The only broad `except` is in `load_checkpoint` (torch raises many
  exception types when reading files); it re-raises `CheckpointError` with the original cause chained.
- OOM handling can only be exercised for real on a device that runs out of memory. The tests simulate
  it by patching the forward pass, which verifies the retry logic but not actual allocator behaviour.

## Workflow when training fails

1. Read the exception message; it names the epoch and step.
2. Run `scripts/train.py --config configs/quick_test.yaml` to see whether the failure is
   configuration-specific.
3. Check `train.log` for gradient diagnostics logged before the failure.
4. Lower `learning_rate`, keep `max_grad_norm` enabled and verify inputs are finite
   (`validate_dataframe` already rejects Inf; NaN features are imputed).
5. Re-run with a fixed seed so the failure can be reproduced.

## Serving errors

| HTTP status | Meaning |
|---|---|
| 200 | Success |
| 413 | Batch larger than `inference.max_api_batch_size` |
| 422 | Invalid payload (schema violation or wrong feature count) |
| 500 | Library error during inference, message included in `detail` |
| 503 | Model not loaded (checkpoint missing or invalid at start-up; see server log) |