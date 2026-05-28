"""Unit tests for early stopping utility."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.training.early_stopper import EarlyStopper


def test_no_stop_on_improvement():
    stopper = EarlyStopper(patience=3, mode="min")
    assert stopper.should_stop(1.0) is False
    assert stopper.should_stop(0.9) is False
    assert stopper.should_stop(0.8) is False


def test_stops_after_patience():
    stopper = EarlyStopper(patience=3, mode="min")
    stopper.should_stop(1.0)
    stopper.should_stop(1.0)
    stopper.should_stop(1.0)
    assert stopper.should_stop(1.0) is True


def test_max_mode_stops_on_no_improvement():
    stopper = EarlyStopper(patience=2, mode="max")
    stopper.should_stop(0.8)
    stopper.should_stop(0.7)
    assert stopper.should_stop(0.7) is True


def test_reset_clears_state():
    stopper = EarlyStopper(patience=2, mode="min")
    stopper.should_stop(1.0)
    stopper.should_stop(1.0)
    stopper.reset()
    assert stopper.best is None
    assert stopper.counter == 0


def test_best_value_tracks_minimum():
    stopper = EarlyStopper(patience=3, mode="min")
    stopper.should_stop(1.0)
    stopper.should_stop(0.5)
    stopper.should_stop(0.8)
    assert stopper.best == 0.5


def test_patience_remaining():
    stopper = EarlyStopper(patience=5, mode="min")
    stopper.should_stop(1.0)
    stopper.should_stop(1.0)
    assert stopper.patience_remaining == 3
