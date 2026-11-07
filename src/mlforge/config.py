from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from mlforge.exceptions import ConfigError

VALID_ACTIVATIONS = ("relu", "tanh", "gelu", "leaky_relu", "silu")
VALID_OPTIMIZERS = ("adam", "adamw", "sgd")
VALID_SCHEDULERS = ("none", "plateau")
MIN_SYNTHETIC_SAMPLES = 50
CLUSTERS_PER_CLASS = 2


@dataclass
class DataConfig:
    csv_path: str | None = None
    target_column: str = "target"
    n_samples: int = 6000
    n_features: int = 20
    n_informative: int = 10
    n_redundant: int = 4
    n_classes: int = 2
    class_sep: float = 1.0
    flip_y: float = 0.02
    missing_fraction: float = 0.02
    max_missing_fraction: float = 0.5
    val_size: float = 0.15
    test_size: float = 0.15

    def __post_init__(self) -> None:
        if self.n_samples < MIN_SYNTHETIC_SAMPLES:
            raise ConfigError(f"data.n_samples must be >= {MIN_SYNTHETIC_SAMPLES}.")
        if self.n_informative + self.n_redundant > self.n_features:
            raise ConfigError("data.n_informative + data.n_redundant must be <= data.n_features.")
        if self.n_classes < 2:
            raise ConfigError("data.n_classes must be >= 2.")
        if self.n_classes * CLUSTERS_PER_CLASS > 2**self.n_informative:
            raise ConfigError("data.n_informative is too small for data.n_classes.")
        if not 0.0 <= self.missing_fraction < 1.0:
            raise ConfigError("data.missing_fraction must be in [0, 1).")
        if not 0.0 < self.max_missing_fraction <= 1.0:
            raise ConfigError("data.max_missing_fraction must be in (0, 1].")
        if self.val_size <= 0 or self.test_size <= 0 or self.val_size + self.test_size >= 1:
            raise ConfigError("data.val_size and data.test_size must be > 0 and sum to < 1.")


@dataclass
class ModelConfig:
    hidden_dims: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    dropout: float = 0.1

    def __post_init__(self) -> None:
        self.hidden_dims = tuple(int(h) for h in self.hidden_dims)
        if any(h < 1 for h in self.hidden_dims):
            raise ConfigError("model.hidden_dims must contain positive integers.")
        if self.activation not in VALID_ACTIVATIONS:
            raise ConfigError(f"model.activation must be one of {VALID_ACTIVATIONS}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1).")


@dataclass
class BaselineConfig:
    C: float = 1.0
    max_iter: int = 500

    def __post_init__(self) -> None:
        if self.C <= 0 or self.max_iter < 1:
            raise ConfigError("baseline.C must be > 0 and baseline.max_iter >= 1.")


@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 128
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    max_grad_norm: float | None = 5.0
    scheduler: str = "none"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    num_workers: int = 0
    pin_memory: bool | str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ConfigError("training.epochs and training.batch_size must be >= 1.")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ConfigError(f"training.optimizer must be one of {VALID_OPTIMIZERS}.")
        if self.learning_rate <= 0:
            raise ConfigError("training.learning_rate must be > 0.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ConfigError("training.max_grad_norm must be > 0 or null.")
        if self.scheduler not in VALID_SCHEDULERS:
            raise ConfigError(f"training.scheduler must be one of {VALID_SCHEDULERS}.")
        if self.num_workers < 0 or self.early_stopping_patience < 0:
            raise ConfigError("training.num_workers and early_stopping_patience must be >= 0.")
        if not isinstance(self.pin_memory, bool) and str(self.pin_memory).lower() not in (
            "auto",
            "true",
            "false",
        ):
            raise ConfigError("training.pin_memory must be auto, true or false.")


@dataclass
class DiagnosticsConfig:
    gradient_explosion_threshold: float = 1000.0
    gradient_vanishing_threshold: float = 1e-7


@dataclass
class InferenceConfig:
    device: str = "auto"
    batch_size: int = 256
    use_inference_mode: bool = True
    use_amp: bool = False
    adaptive_batch_size: bool = True
    min_batch_size: int = 1
    max_api_batch_size: int = 1024

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.min_batch_size < 1 or self.max_api_batch_size < 1:
            raise ConfigError("inference batch sizes must be >= 1.")
        if self.min_batch_size > self.batch_size:
            raise ConfigError("inference.min_batch_size must be <= inference.batch_size.")


@dataclass
class BenchmarkConfig:
    warmup_iterations: int = 5
    iterations: int = 30
    batch_sizes: tuple[int, ...] = (1, 16, 64, 256)
    num_samples: int = 1024
    devices: tuple[str, ...] = ("cpu", "cuda")
    rtol: float = 1e-4
    atol: float = 1e-4

    def __post_init__(self) -> None:
        self.batch_sizes = tuple(int(b) for b in self.batch_sizes)
        self.devices = tuple(str(d) for d in self.devices)
        if self.warmup_iterations < 0 or self.iterations < 1:
            raise ConfigError("benchmark.warmup_iterations >= 0 and iterations >= 1 required.")
        if not self.batch_sizes or any(b < 1 for b in self.batch_sizes):
            raise ConfigError("benchmark.batch_sizes must be non-empty positive integers.")
        if self.num_samples < max(self.batch_sizes):
            raise ConfigError("benchmark.num_samples must be >= the largest batch size.")


_SECTION_TYPES: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "baseline": BaselineConfig,
    "training": TrainingConfig,
    "diagnostics": DiagnosticsConfig,
    "inference": InferenceConfig,
    "benchmark": BenchmarkConfig,
}


@dataclass
class ExperimentConfig:
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    output_dir: str = "outputs"
    experiment_name: str = "mlforge_default"
    log_level: str = "INFO"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> ExperimentConfig:
        raw = dict(raw or {})
        kwargs: dict[str, Any] = {}
        for name, section_cls in _SECTION_TYPES.items():
            kwargs[name] = _build_section(section_cls, raw.pop(name, None), name)
        valid_top = {f.name for f in fields(cls)} - set(_SECTION_TYPES)
        unknown = set(raw) - valid_top
        if unknown:
            raise ConfigError(f"Unknown top-level config keys: {sorted(unknown)}")
        return cls(**kwargs, **raw)


def _build_section(section_cls: type, raw: dict[str, Any] | None, name: str) -> Any:
    if raw is None:
        return section_cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"Config section '{name}' must be a mapping.")
    valid = {f.name for f in fields(section_cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(f"Unknown keys in config section '{name}': {sorted(unknown)}")
    return section_cls(**raw)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if raw is not None and not isinstance(raw, dict):
        raise ConfigError(f"Top level of {config_path} must be a mapping.")
    return ExperimentConfig.from_dict(raw)