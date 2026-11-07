"""Early stopping on a metric that should decrease."""

from __future__ import annotations


class EarlyStopping:
    """Stops when the metric fails to improve by ``min_delta`` for ``patience`` checks.

    ``patience == 0`` disables stopping (improvement tracking still works).
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Record a new value; return True if it is a new best."""
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.patience > 0 and self.counter >= self.patience:
                self.should_stop = True
        return improved