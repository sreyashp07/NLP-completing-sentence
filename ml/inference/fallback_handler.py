"""
Fallback handler for low-confidence or failed predictions.

When the model is uncertain, route to human review
instead of making a potentially wrong auto-routing decision.
"""
from typing import Dict, Optional


FALLBACK_RESPONSE = {
    "primary_intent": {
        "intent": "general_inquiry",
        "confidence": 0.0,
        "department": "General Support",
        "priority": "low",
        "display_label": "General Inquiry",
    },
    "top_intents": [],
    "keywords": [],
    "model_used": "fallback",
    "processing_time_ms": 0.0,
    "fallback": True,
    "fallback_reason": "unknown",
}


def get_fallback_response(reason: str = "prediction_failed") -> Dict:
    """Return a safe fallback response when prediction fails."""
    response = dict(FALLBACK_RESPONSE)
    response["fallback_reason"] = reason
    return response


def handle_low_confidence(
    prediction: Dict,
    threshold: float = 0.5,
) -> Dict:
    """
    If confidence is too low, override to human review queue.
    Prevents wrong auto-routing on uncertain predictions.
    """
    conf = prediction.get("primary_intent", {}).get("confidence", 0)
    if conf < threshold:
        enhanced = dict(prediction)
        enhanced["requires_human_review"] = True
        enhanced["human_review_reason"] = f"Confidence {conf:.2f} below threshold {threshold}"
        enhanced["primary_intent"]["priority"] = "medium"
        enhanced["primary_intent"]["department"] = "General Support"
        return enhanced
    return prediction


def is_valid_prediction(prediction: Dict) -> bool:
    """Validate that a prediction result has all required fields."""
    if not prediction:
        return False
    primary = prediction.get("primary_intent", {})
    required = ["intent", "confidence", "department", "priority"]
    return all(k in primary for k in required)
