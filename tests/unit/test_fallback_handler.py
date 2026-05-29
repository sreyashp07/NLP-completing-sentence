"""Unit tests for fallback handler."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.inference.fallback_handler import (
    get_fallback_response,
    handle_low_confidence,
    is_valid_prediction,
)


def test_fallback_response_has_required_keys():
    result = get_fallback_response("test_reason")
    assert "primary_intent" in result
    assert result["fallback"] is True
    assert result["fallback_reason"] == "test_reason"


def test_fallback_routes_to_general_support():
    result = get_fallback_response()
    assert result["primary_intent"]["department"] == "General Support"


def test_low_confidence_triggers_review():
    pred = {
        "primary_intent": {
            "intent": "payment_issue",
            "confidence": 0.3,
            "department": "Billing Team",
            "priority": "critical",
        }
    }
    result = handle_low_confidence(pred, threshold=0.6)
    assert result.get("requires_human_review") is True


def test_high_confidence_passes_through():
    pred = {
        "primary_intent": {
            "intent": "payment_issue",
            "confidence": 0.9,
            "department": "Billing Team",
            "priority": "critical",
        }
    }
    result = handle_low_confidence(pred, threshold=0.6)
    assert result.get("requires_human_review") is None


def test_is_valid_prediction_true():
    pred = {
        "primary_intent": {
            "intent": "payment_issue",
            "confidence": 0.9,
            "department": "Billing Team",
            "priority": "critical",
        }
    }
    assert is_valid_prediction(pred) is True


def test_is_valid_prediction_false():
    assert is_valid_prediction({}) is False
    assert is_valid_prediction(None) is False
