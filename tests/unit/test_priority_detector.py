"""Unit tests for priority escalation detector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.priority_detector import (
    detect_priority_signals,
    should_override_priority,
)


def test_fraud_triggers_critical():
    result = detect_priority_signals("I think this is a fraud and scam")
    assert result["recommended_priority"] == "critical"
    assert result["should_escalate"] is True


def test_urgent_triggers_high():
    result = detect_priority_signals("urgent help needed immediately")
    assert result["recommended_priority"] in ["high", "critical"]


def test_normal_text_no_override():
    result = detect_priority_signals("what are your support hours")
    assert result["recommended_priority"] is None
    assert result["should_escalate"] is False


def test_escalation_phrase_detected():
    result = detect_priority_signals("I am taking legal action against you")
    assert len(result["escalation_phrases_found"]) > 0
    assert result["recommended_priority"] == "critical"


def test_should_override_low_to_critical():
    final, overridden = should_override_priority("low", "this is fraud and scam")
    assert final == "critical"
    assert overridden is True


def test_no_override_when_already_critical():
    final, overridden = should_override_priority("critical", "help me please")
    assert final == "critical"
    assert overridden is False
