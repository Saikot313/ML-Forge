# Architecture

## Layered view

| Layer | Modules | Responsibility |
|---|---|---|
| Foundation | `exceptions`, `logging_utils`, `config`, `seed`, `device` | Shared types, logging, validated configuration, seeding, device resolution |
| Data | `data/*` | Loading, validation, preprocessing, splitting, DataLoaders |
| Models | `models/*` | MLP and Logistic Regression baseline |
| Numerics | `numerics/*` | Stability, linear algebra, gradient utilities |
| Observability | `memory/*`, `diagnostics/*` | Memory measurement/estimation, failure detection |
| Persistence | `checkpointing` | Atomic checkpoints, validation, model restore |
| Workflows | `training/*`, `evaluation/*` | Trainer, pipeline, metrics, comparison |
| Serving | `inference/*`, `api/*` | Predictor and FastAPI app |
| Measurement | `benchmarking/*` | Latency/throughput benchmark, DataLoader benchmark, profiler |

## Dependency direction

Dependencies point downward only:

```
scripts -> training.pipeline, benchmarking, api, inference
training.pipeline -> trainer, evaluation, inference.predictor, checkpointing, data, models
inference.predictor -> checkpointing, data.preprocessing, diagnostics, numerics
checkpointing -> diagnostics, models
diagnostics -> numerics
```

`inference` never imports `training`, because checkpoint code lives in `mlforge/checkpointing.py`.
`mlforge.training.__init__` does not import `pipeline`, so importing the training package cannot pull
in inference.

## Data flow

1. `load_config` builds a typed `ExperimentConfig` (unknown keys rejected).
2. `prepare_data` loads a CSV or generates synthetic data, validates it, splits it (stratified,
   seeded), fits `TabularPreprocessor` on the **training split only**, and transforms all splits.
3. `Trainer.fit` trains the MLP, tracks history, selects the best epoch by validation loss, saves the
   checkpoint at each improvement, and restores the best weights at the end.
4. `run_training` reloads the checkpoint through `Predictor`, scores the test split for the MLP and
   baseline with the same metric function, and checks the reloaded model's outputs against the
   in-memory model.
5. `Predictor`, the benchmark scripts and the API all consume the checkpoint.

## Checkpoint contents

`version`, `model_state_dict`, `model_init_config`, `optimizer_state_dict`, `epoch`, `best_val_loss`,
`history`, `config`, `preprocessor` (plain lists), `feature_names`, `metrics`, `metadata` (UTC time,
Python/torch/numpy versions, platform, seed, device). Only tensors and plain Python types are stored,
so loading uses `torch.load(weights_only=True)`. Writes are atomic (temp file, then `os.replace`).

## Design decisions

- **Dataclass config over a config library:** minimal dependencies and strict validation.
- **NumPy preprocessor instead of a pickled scikit-learn transformer:** state is plain lists in the
  checkpoint, which is safe to load and easy to inspect.
- **Exceptions hierarchy:** `DataValidationError` maps to HTTP 422 in the API; other `MLForgeError`s
  map to 500.
- **No MLflow:** local JSON metadata is sufficient at this scale.
- **GPU is always conditional:** requesting CUDA when it is unavailable raises `DeviceError`; the
  benchmark skips CUDA silently with a log message; `use_amp` is ignored on CPU with a warning.

## Adding a model

Implement a module with a forward pass producing `(batch, n_classes)` logits and an `init_config()` /
`from_init_config()` pair, then extend `checkpointing.restore_model` and `Trainer` typing. The
metric, benchmark and API code operate on probabilities or logits and need no changes.