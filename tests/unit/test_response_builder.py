"""Unit tests for response builder utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.utils.response_builder import (
    build_success,
    build_error,
    build_prediction_summary,
    build_batch_summary,
)


def test_build_success_structure():
    result = build_success({"key": "value"})
    assert result["success"] is True
    assert "timestamp" in result
    assert result["data"] == {"key": "value"}


def test_build_error_structure():
    result = build_error("Something failed", code=404)
    assert result["success"] is False
    assert result["code"] == 404
    assert "timestamp" in result


def test_build_prediction_summary():
    pred = {
        "primary_intent": {
            "intent": "payment_issue",
            "confidence": 0.914,
            "department": "Billing Team",
            "priority": "critical",
        },
        "processing_time_ms": 12.4,
    }
    result = build_prediction_summary(pred)
    assert result["intent"] == "payment_issue"
    assert result["confidence"] == 0.914
    assert result["priority"] == "critical"


def test_build_batch_summary_empty():
    result = build_batch_summary([])
    assert result["total"] == 0


def test_build_batch_summary_counts():
    preds = [
        {"primary_intent": {"intent": "payment_issue", "priority": "critical", "confidence": 0.9}},
        {"primary_intent": {"intent": "payment_issue", "priority": "critical", "confidence": 0.8}},
        {"primary_intent": {"intent": "refund_request", "priority": "high", "confidence": 0.7}},
    ]
    result = build_batch_summary(preds)
    assert result["total"] == 3
    assert result["critical_count"] == 2
    assert result["intent_distribution"]["payment_issue"] == 2
