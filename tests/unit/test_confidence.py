"""Unit tests for confidence calibration utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.inference.confidence_calibrator import (
    apply_confidence_threshold,
    get_confidence_label,
    filter_low_confidence_predictions,
)


def test_high_confidence_passes():
    meets, label = apply_confidence_threshold(0.95)
    assert meets is True
    assert label == "high_confidence"


def test_low_confidence_fails():
    meets, label = apply_confidence_threshold(0.3)
    assert meets is False


def test_confidence_label_very_high():
    assert get_confidence_label(0.95) == "Very High"


def test_confidence_label_low():
    assert get_confidence_label(0.45) == "Low"


def test_filter_predictions():
    preds = [
        {"primary_intent": {"confidence": 0.9}},
        {"primary_intent": {"confidence": 0.3}},
        {"primary_intent": {"confidence": 0.7}},
    ]
    confident, uncertain = filter_low_confidence_predictions(preds)
    assert len(confident) == 2
    assert len(uncertain) == 1
