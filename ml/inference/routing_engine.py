"""
Advanced routing engine for ticket assignment.

Combines intent classification + sentiment analysis
+ confidence scoring to make smarter routing decisions.
"""
from typing import Dict
from ml.preprocessing.sentiment_analyzer import analyze_sentiment

PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"

ROUTING_QUEUES = {
    "immediate_escalation": {
        "description": "Critical issue + frustrated customer",
        "sla_minutes": 15,
        "requires_senior": True,
    },
    "fast_track": {
        "description": "High priority or urgent sentiment",
        "sla_minutes": 60,
        "requires_senior": False,
    },
    "human_review": {
        "description": "Low confidence prediction needs human",
        "sla_minutes": 120,
        "requires_senior": False,
    },
    "standard": {
        "description": "Normal routing to department",
        "sla_minutes": 480,
        "requires_senior": False,
    },
    "low_priority": {
        "description": "Low urgency, can be batched",
        "sla_minutes": 2880,
        "requires_senior": False,
    },
}


def determine_queue(
    prediction: Dict,
    confidence_threshold: float = 0.6,
) -> Dict:
    """
    Determine which routing queue a ticket belongs to.

    Args:
        prediction: Full prediction result dict
        confidence_threshold: Minimum confidence for auto-routing

    Returns:
        Queue assignment with metadata
    """
    primary = prediction.get("primary_intent", {})
    intent = primary.get("intent", "general_inquiry")
    priority = primary.get("priority", "low")
    confidence = primary.get("confidence", 0.0)
    text = prediction.get("text", "")

    sentiment = analyze_sentiment(text)

    if confidence < confidence_threshold:
        queue = "human_review"
        reason = f"Low confidence ({confidence:.2f}) needs human verification"

    elif priority == PRIORITY_CRITICAL and sentiment["sentiment"] == "frustrated":
        queue = "immediate_escalation"
        reason = "Critical intent with frustrated customer sentiment"

    elif priority in [PRIORITY_CRITICAL, PRIORITY_HIGH] or sentiment["is_urgent"]:
        queue = "fast_track"
        reason = "High priority intent or urgent sentiment detected"

    elif priority == "low" and not sentiment["is_urgent"]:
        queue = "low_priority"
        reason = "Low priority non-urgent ticket"

    else:
        queue = "standard"
        reason = "Standard department routing"

    queue_info = ROUTING_QUEUES[queue]

    return {
        "queue": queue,
        "reason": reason,
        "sla_minutes": queue_info["sla_minutes"],
        "requires_senior": queue_info["requires_senior"],
        "sentiment": sentiment["sentiment"],
        "is_urgent": sentiment["is_urgent"],
        "confidence": confidence,
        "department": primary.get("department", "General Support"),
        "priority": priority,
    }
