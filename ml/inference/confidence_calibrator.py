"""
Confidence calibration utilities.

Raw model probabilities can be overconfident or underconfident.
This module provides utilities to assess and adjust confidence scores.
"""
from typing import List, Tuple


def apply_confidence_threshold(
    confidence: float,
    threshold: float = 0.6,
) -> Tuple[bool, str]:
    """
    Check if confidence meets the minimum threshold.

    Args:
        confidence: Raw model confidence score
        threshold: Minimum acceptable confidence

    Returns:
        Tuple of (meets_threshold, recommendation)
    """
    if confidence >= 0.9:
        return True, "high_confidence"
    elif confidence >= threshold:
        return True, "acceptable_confidence"
    elif confidence >= 0.4:
        return False, "low_confidence_review"
    else:
        return False, "very_low_confidence_escalate"


def get_confidence_label(confidence: float) -> str:
    """Return human-readable confidence label."""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.75:
        return "High"
    elif confidence >= 0.6:
        return "Medium"
    elif confidence >= 0.4:
        return "Low"
    else:
        return "Very Low"


def filter_low_confidence_predictions(
    predictions: List[dict],
    threshold: float = 0.6,
) -> Tuple[List[dict], List[dict]]:
    """
    Split predictions into confident and uncertain groups.

    Returns:
        Tuple of (confident_predictions, uncertain_predictions)
    """
    confident = []
    uncertain = []
    for pred in predictions:
        conf = pred.get("primary_intent", {}).get("confidence", 0)
        if conf >= threshold:
            confident.append(pred)
        else:
            uncertain.append(pred)
    return confident, uncertain
