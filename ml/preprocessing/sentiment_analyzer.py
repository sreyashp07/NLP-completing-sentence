"""
Lightweight sentiment analysis for customer support messages.

Detects emotional tone to help prioritize urgent or
frustrated customer tickets automatically.

Sentiment categories:
- frustrated: angry, upset customer needs fast response
- neutral: standard inquiry
- positive: happy customer, low urgency
"""
import re
from typing import Dict

FRUSTRATED_WORDS = {
    "angry", "furious", "horrible", "terrible", "worst", "awful",
    "ridiculous", "unacceptable", "disgusting", "pathetic", "useless",
    "scam", "fraud", "cheated", "lied", "stolen", "robbery",
    "disappointed", "frustrated", "fed up", "sick", "done",
    "never again", "hate", "rubbish", "garbage", "waste",
    "incompetent", "unprofessional", "disgrace",
}

URGENT_WORDS = {
    "urgent", "asap", "immediately", "emergency", "critical",
    "right now", "today", "tonight", "deadline", "expire",
    "blocked", "stuck", "cannot", "unable", "failed", "error",
}

POSITIVE_WORDS = {
    "thank", "thanks", "appreciate", "happy", "satisfied",
    "great", "wonderful", "excellent", "love", "perfect",
    "good", "nice", "helpful", "pleased",
}


def analyze_sentiment(text: str) -> Dict:
    """
    Analyze sentiment of a customer support message.

    Args:
        text: Customer message

    Returns:
        Dict with sentiment, urgency, and scores
    """
    text_lower = text.lower()
    words = set(text_lower.split())

    frustrated_score = len(words & FRUSTRATED_WORDS)
    urgent_score = len(words & URGENT_WORDS)
    positive_score = len(words & POSITIVE_WORDS)

    exclamation_count = text.count("!")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    frustrated_score += exclamation_count * 0.5
    frustrated_score += caps_ratio * 3

    if frustrated_score >= 2:
        sentiment = "frustrated"
    elif positive_score >= 2 and frustrated_score == 0:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    is_urgent = urgent_score >= 1 or exclamation_count >= 2

    return {
        "sentiment": sentiment,
        "is_urgent": is_urgent,
        "frustrated_score": round(frustrated_score, 2),
        "urgent_score": urgent_score,
        "positive_score": positive_score,
        "exclamation_count": exclamation_count,
        "caps_ratio": round(caps_ratio, 3),
    }
