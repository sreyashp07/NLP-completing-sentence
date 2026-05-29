"""
Intent result formatting utilities.

Converts raw prediction dicts into human-readable
strings for display in UI, emails, and API responses.
"""
from typing import Dict


PRIORITY_LABELS = {
    "critical": "🚨 CRITICAL — Immediate action required",
    "high":     "🔴 HIGH — Respond within 4 hours",
    "medium":   "🟡 MEDIUM — Respond within 24 hours",
    "low":      "🟢 LOW — Respond within 72 hours",
}


def format_intent_for_agent(prediction: Dict) -> str:
    """
    Format prediction result as a readable summary for support agents.

    Example output:
    ┌─────────────────────────────────┐
    │ Intent: Payment Issue (91.4%)   │
    │ Route to: Billing Team          │
    │ Priority: CRITICAL              │
    │ Keywords: payment, failed       │
    └─────────────────────────────────┘
    """
    primary = prediction.get("primary_intent", {})
    intent = primary.get("display_label", "Unknown")
    conf = primary.get("confidence", 0) * 100
    dept = primary.get("department", "General Support")
    priority = primary.get("priority", "low")
    keywords = prediction.get("keywords", [])

    priority_label = PRIORITY_LABELS.get(priority, priority)

    lines = [
        "─" * 40,
        f"Intent:    {intent} ({conf:.1f}%)",
        f"Route to:  {dept}",
        f"Priority:  {priority_label}",
    ]
    if keywords:
        lines.append(f"Keywords:  {', '.join(keywords[:4])}")
    lines.append("─" * 40)

    return "\n".join(lines)


def format_intent_for_email(prediction: Dict) -> str:
    """Format prediction as email-friendly text."""
    primary = prediction.get("primary_intent", {})
    return (
        f"This ticket has been automatically classified as "
        f"'{primary.get('display_label', 'Unknown')}' "
        f"with {primary.get('confidence', 0)*100:.0f}% confidence "
        f"and routed to {primary.get('department', 'General Support')}."
    )


def format_confidence_bar(confidence: float, width: int = 20) -> str:
    """Render ASCII confidence bar."""
    filled = int(confidence * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"|{bar}| {confidence*100:.1f}%"
