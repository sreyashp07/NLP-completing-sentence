"""Unit tests for advanced routing engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.inference.routing_engine import determine_queue, ROUTING_QUEUES

def make_prediction(intent, priority, confidence, text="help me"):
    return {
        "text": text,
        "primary_intent": {
            "intent": intent,
            "priority": priority,
            "confidence": confidence,
            "department": "Billing Team",
            "display_label": intent,
        }
    }

def test_low_confidence_goes_to_human_review():
    pred = make_prediction("payment_issue", "critical", 0.3)
    result = determine_queue(pred)
    assert result["queue"] == "human_review"

def test_critical_frustrated_escalates():
    pred = make_prediction(
        "payment_issue", "critical", 0.95,
        "This is the WORST service ever!!! Terrible fraud!"
    )
    result = determine_queue(pred)
    assert result["queue"] == "immediate_escalation"

def test_high_priority_fast_track():
    pred = make_prediction("technical_bug", "high", 0.85)
    result = determine_queue(pred)
    assert result["queue"] in ["fast_track", "standard"]

def test_low_priority_non_urgent():
    pred = make_prediction("feature_request", "low", 0.9)
    result = determine_queue(pred)
    assert result["queue"] in ["low_priority", "standard"]

def test_result_has_required_keys():
    pred = make_prediction("general_inquiry", "low", 0.8)
    result = determine_queue(pred)
    keys = ["queue", "reason", "sla_minutes", "sentiment", "confidence"]
    for key in keys:
        assert key in result

def test_all_queues_have_sla():
    for queue_name, info in ROUTING_QUEUES.items():
        assert "sla_minutes" in info
        assert info["sla_minutes"] > 0
