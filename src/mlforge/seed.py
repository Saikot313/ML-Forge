from __future__ import annotations

import os
import random

import numpy as np
import torch

from mlforge.logging_utils import get_logger

logger = get_logger(__name__)
_MAX_WORKER_SEED = 2**32


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects only child processes

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
    logger.info("Global seed set to %d (deterministic=%s).", seed, deterministic)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % _MAX_WORKER_SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)