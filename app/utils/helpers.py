"""
Utility helper functions used across the application.
"""
from typing import Dict, List
from datetime import datetime


def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage string."""
    return f"{confidence * 100:.1f}%"


def get_priority_color(priority: str) -> str:
    """Return hex color for priority level."""
    colors = {
        "critical": "#FF2D55",
        "high": "#FF6B35",
        "medium": "#FFD700",
        "low": "#34C759",
    }
    return colors.get(priority, "#888888")


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis if too long."""
    return text[:max_length] + "..." if len(text) > max_length else text


def generate_ticket_id(index: int) -> str:
    """Generate a ticket ID from index."""
    return f"TKT-{index + 1001:04d}"


def get_timestamp() -> str:
    """Return current timestamp as formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
