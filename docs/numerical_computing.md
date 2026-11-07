# Numerical Computing

Every utility here has a verification test in `tests/test_numerics.py`.

## Naive versus stable

| Naive | Stable | Problem the naive version has |
|---|---|---|
| `naive_softmax` | `stable_softmax` | `exp(x)` overflows to inf for large inputs, giving inf/inf = NaN |
| `naive_sigmoid` | `stable_sigmoid` | `exp(-x)` overflows for large negative `x` (RuntimeWarning) |
| softmax then log | `stable_log_softmax_t` | `log(0)` when a probability underflows |
| – | `logsumexp` | Max-shift avoids overflow in `log(sum(exp(x)))` |
| divide by norm | `safe_normalize` | Zero vectors would produce NaN; here they map to zeros |

`cross_entropy_from_logits` computes the loss from stable log-probabilities. A test checks it matches
`torch.nn.functional.cross_entropy` and stays finite for logits of magnitude 1e4. The model's own
training loss uses `F.cross_entropy` on raw logits, which is already stable. The custom function exists
as an inspectable reference, not as a replacement.

## Finiteness checks

`count_nonfinite` counts NaN and Inf in arrays or tensors (integer arrays return zeros).
`assert_finite` raises `NumericalInstabilityError` with the counts.

## Tolerance-based comparison

`compare_arrays(a, b, rtol, atol)` returns a `ToleranceReport` (`allclose`, max absolute and relative
difference, number of mismatched elements). NaN never compares equal, so a NaN-producing
implementation cannot pass a comparison by accident. The benchmark, the checkpoint round-trip check and
the tests use this one function.

Tolerance guidance: fp32 forward passes compared across code paths on the same device generally need
`rtol` around 1e-5; comparisons against fp16 autocast outputs need looser tolerances and may
legitimately fail strict ones. The benchmark records `matches_reference` per row instead of assuming it.

## Linear algebra

- `matmul_reference` (triple loop) versus `matmul_vectorized` (BLAS) with `verify_matmul`.
- `manual_linear_forward` / `verify_linear_layer`: `y = x @ W.T + b` compared with `nn.Linear`.
- `pairwise_squared_distances` uses the expansion `|a|² + |b|² − 2ab`, with O(n·m) memory, instead of
  the O(n·m·d) broadcast tensor in `pairwise_squared_distances_broadcast`. Rounding can produce tiny
  negative values, which are clamped to zero.
- `standardize_columns` handles constant columns without dividing by zero.

## Gradients

`gradient_norm` is the global L2 norm over parameters that have gradients. `clip_gradients` scales
gradients in place when the norm exceeds `max_norm` and returns the pre-clipping norm. It raises
`NumericalInstabilityError` if the norm is NaN or Inf, because scaling cannot repair non-finite
gradients. A test compares its result with `torch.nn.utils.clip_grad_norm_`.

## Where this is used in the pipeline

The preprocessor (median imputation and safe standardisation), the Predictor's softmax, the trainer's
gradient handling, the metric normalisation, and the benchmark's correctness check all rely on these
utilities.