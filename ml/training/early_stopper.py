"""
Early stopping utility for model training.

Monitors validation metric and stops training
when performance stops improving.

Used in transformer model training (Phase 2)
to prevent overfitting on customer support data.
"""
from typing import Optional


class EarlyStopper:
    """
    Early stopping callback for training loops.

    Usage:
        stopper = EarlyStopper(patience=3, min_delta=0.001)
        for epoch in range(100):
            val_loss = train_epoch()
            if stopper.should_stop(val_loss):
                break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """
        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum improvement to count as progress
            mode: 'min' for loss, 'max' for accuracy/F1
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value: Optional[float] = None
        self.counter = 0
        self.stopped_epoch = 0

    def should_stop(self, current_value: float) -> bool:
        """
        Check if training should stop.

        Returns True if no improvement for `patience` epochs.
        """
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == "min":
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset the early stopper state."""
        self.best_value = None
        self.counter = 0
        self.stopped_epoch = 0

    @property
    def best(self) -> Optional[float]:
        """Return the best value seen so far."""
        return self.best_value

    @property
    def patience_remaining(self) -> int:
        """Return remaining patience epochs."""
        return self.patience - self.counter
