"""
Intent aggregator for multi-message analysis.

When a customer sends multiple messages in a conversation,
this module aggregates predictions to determine the
overall dominant intent and priority.
"""
from typing import Dict, List
from collections import Counter


def aggregate_predictions(predictions: List[Dict]) -> Dict:
    """
    Aggregate multiple predictions into a single dominant result.

    Args:
        predictions: List of prediction dicts from IntentRouter

    Returns:
        Aggregated result with dominant intent and priority
    """
    if not predictions:
        return {}

    if len(predictions) == 1:
        return predictions[0]

    intents = []
    priorities = []
    confidences = []

    for pred in predictions:
        primary = pred.get("primary_intent", {})
        intents.append(primary.get("intent", "general_inquiry"))
        priorities.append(primary.get("priority", "low"))
        confidences.append(primary.get("confidence", 0.0))

    intent_counts = Counter(intents)
    dominant_intent = intent_counts.most_common(1)[0][0]

    priority_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    highest_priority = max(priorities, key=lambda p: priority_rank.get(p, 0))

    avg_confidence = sum(confidences) / len(confidences)

    dominant_pred = next(
        (p for p in predictions
         if p.get("primary_intent", {}).get("intent") == dominant_intent),
        predictions[0]
    )

    return {
        "dominant_intent": dominant_intent,
        "highest_priority": highest_priority,
        "avg_confidence": round(avg_confidence, 4),
        "message_count": len(predictions),
        "intent_distribution": dict(intent_counts),
        "primary_intent": {
            **dominant_pred.get("primary_intent", {}),
            "priority": highest_priority,
        },
    }
