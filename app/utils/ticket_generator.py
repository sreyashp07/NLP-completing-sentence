"""
Ticket ID and metadata generator.

Generates unique ticket IDs, timestamps and
metadata for classified support tickets.
"""
import uuid
import hashlib
from datetime import datetime
from typing import Dict


def generate_ticket_id(prefix: str = "TKT") -> str:
    """Generate a unique ticket ID with prefix."""
    unique = str(uuid.uuid4())[:8].upper()
    return f"{prefix}-{unique}"


def generate_short_id(index: int) -> str:
    """Generate sequential short ticket ID."""
    return f"TKT-{index + 1001:04d}"


def generate_ticket_hash(text: str, timestamp: str) -> str:
    """Generate a deterministic hash for deduplication."""
    raw = f"{text.strip().lower()}:{timestamp}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def create_ticket_metadata(
    text: str,
    prediction: Dict,
    ticket_index: int,
) -> Dict:
    """
    Create full ticket metadata from a prediction result.

    Args:
        text: Original customer message
        prediction: Prediction result dict
        ticket_index: Sequential ticket number

    Returns:
        Complete ticket metadata dict
    """
    now = datetime.now()
    primary = prediction.get("primary_intent", {})

    return {
        "id": generate_short_id(ticket_index),
        "hash": generate_ticket_hash(text, now.isoformat()),
        "created_at": now.isoformat(),
        "date": now.strftime("%b %d"),
        "timestamp": now.strftime("%H:%M:%S"),
        "text": text[:200],
        "intent": primary.get("intent", "unknown"),
        "display_label": primary.get("display_label", "Unknown"),
        "department": primary.get("department", "General Support"),
        "priority": primary.get("priority", "low"),
        "confidence": primary.get("confidence", 0.0),
        "status": "open",
        "model_used": prediction.get("model_used", "baseline"),
    }
