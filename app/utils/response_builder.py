"""
API response builder utilities.

Provides consistent response construction helpers
so all API endpoints return the same structure.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def build_success(
    data: Any,
    message: str = "Request processed successfully",
) -> Dict:
    """Build a standard success response."""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_error(
    message: str,
    code: int = 500,
    details: Optional[str] = None,
) -> Dict:
    """Build a standard error response."""
    return {
        "success": False,
        "error": message,
        "code": code,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def build_prediction_summary(prediction: Dict) -> Dict:
    """Build a condensed prediction summary for logging."""
    primary = prediction.get("primary_intent", {})
    return {
        "intent": primary.get("intent"),
        "confidence": round(primary.get("confidence", 0), 3),
        "department": primary.get("department"),
        "priority": primary.get("priority"),
        "processing_time_ms": prediction.get("processing_time_ms"),
    }


def build_batch_summary(predictions: List[Dict]) -> Dict:
    """Build a summary of batch prediction results."""
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    intents: Dict = {}
    priorities: Dict = {}
    total_conf = 0.0

    for pred in predictions:
        primary = pred.get("primary_intent", {})
        intent = primary.get("intent", "unknown")
        priority = primary.get("priority", "low")
        conf = primary.get("confidence", 0.0)

        intents[intent] = intents.get(intent, 0) + 1
        priorities[priority] = priorities.get(priority, 0) + 1
        total_conf += conf

    return {
        "total": total,
        "avg_confidence": round(total_conf / total, 4),
        "intent_distribution": intents,
        "priority_distribution": priorities,
        "critical_count": priorities.get("critical", 0),
    }
