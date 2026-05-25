"""
Rule-based priority escalation detector.

Works alongside the ML model to catch
high-urgency signals that need immediate attention.
Rules are based on real support team escalation criteria.
"""
import re
from typing import Dict, List


CRITICAL_KEYWORDS = {
    "fraud", "scam", "hacked", "unauthorized", "stolen",
    "legal", "lawsuit", "police", "fir", "complaint",
    "bank", "dispute", "chargeback", "block", "freeze",
}

HIGH_KEYWORDS = {
    "urgent", "asap", "immediately", "emergency", "deadline",
    "expire", "locked", "suspended", "disabled", "breach",
    "failed", "error", "broken", "crash", "down",
}

ESCALATION_PHRASES = [
    "contact my bank",
    "filing a complaint",
    "taking legal action",
    "reporting to authorities",
    "social media",
    "never using again",
    "want to sue",
    "consumer court",
]


def detect_priority_signals(text: str) -> Dict:
    """
    Detect priority escalation signals in customer message.

    Args:
        text: Customer support message

    Returns:
        Dict with detected signals and recommended priority
    """
    text_lower = text.lower()
    words = set(text_lower.split())

    critical_hits = words & CRITICAL_KEYWORDS
    high_hits = words & HIGH_KEYWORDS

    escalation_hits = [
        phrase for phrase in ESCALATION_PHRASES
        if phrase in text_lower
    ]

    exclamations = text.count("!")
    all_caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]

    if critical_hits or escalation_hits:
        recommended_priority = "critical"
    elif high_hits or exclamations >= 2 or len(all_caps_words) >= 2:
        recommended_priority = "high"
    else:
        recommended_priority = None

    return {
        "recommended_priority": recommended_priority,
        "critical_keywords_found": list(critical_hits),
        "high_keywords_found": list(high_hits),
        "escalation_phrases_found": escalation_hits,
        "exclamation_count": exclamations,
        "caps_word_count": len(all_caps_words),
        "should_escalate": recommended_priority == "critical",
    }


def should_override_priority(
    ml_priority: str,
    text: str,
) -> tuple:
    """
    Check if rule-based signals should override ML priority.

    Returns:
        Tuple of (final_priority, was_overridden)
    """
    signals = detect_priority_signals(text)
    rule_priority = signals["recommended_priority"]

    priority_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    if rule_priority and priority_rank.get(rule_priority, 0) > priority_rank.get(ml_priority, 0):
        return rule_priority, True

    return ml_priority, False
