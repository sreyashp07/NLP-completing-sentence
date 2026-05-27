"""Unit tests for intent aggregator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.inference.intent_aggregator import aggregate_predictions

def make_pred(intent, priority, confidence):
    return {
        "primary_intent": {
            "intent": intent,
            "priority": priority,
            "confidence": confidence,
            "department": "Billing Team",
            "display_label": intent,
        }
    }

def test_empty_predictions():
    result = aggregate_predictions([])
    assert result == {}

def test_single_prediction_passthrough():
    pred = make_pred("payment_issue", "critical", 0.9)
    result = aggregate_predictions([pred])
    assert result == pred

def test_dominant_intent_detected():
    preds = [
        make_pred("payment_issue", "critical", 0.9),
        make_pred("payment_issue", "critical", 0.85),
        make_pred("refund_request", "high", 0.7),
    ]
    result = aggregate_predictions(preds)
    assert result["dominant_intent"] == "payment_issue"

def test_highest_priority_selected():
    preds = [
        make_pred("feature_request", "low", 0.9),
        make_pred("account_locked", "critical", 0.8),
    ]
    result = aggregate_predictions(preds)
    assert result["highest_priority"] == "critical"

def test_message_count_correct():
    preds = [make_pred("payment_issue", "high", 0.8) for _ in range(5)]
    result = aggregate_predictions(preds)
    assert result["message_count"] == 5

def test_avg_confidence_calculated():
    preds = [
        make_pred("payment_issue", "high", 0.8),
        make_pred("payment_issue", "high", 0.6),
    ]
    result = aggregate_predictions(preds)
    assert result["avg_confidence"] == 0.7
