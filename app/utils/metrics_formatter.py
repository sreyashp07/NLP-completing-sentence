"""
Metrics formatting utilities for consistent display.

Used by both API responses and Streamlit UI
to format model metrics consistently.
"""
from typing import Dict, List


def format_metrics_for_display(metrics: Dict) -> Dict:
    """Format raw metrics dict for human-readable display."""
    formatted = {}
    for key, val in metrics.items():
        if isinstance(val, float):
            formatted[key] = f"{val*100:.1f}%"
        else:
            formatted[key] = str(val)
    return formatted


def format_confusion_matrix_row(
    class_name: str,
    correct: int,
    total: int,
) -> str:
    """Format a single confusion matrix row as string."""
    accuracy = correct / total if total > 0 else 0
    bar_length = int(accuracy * 20)
    bar = "█" * bar_length + "░" * (20 - bar_length)
    return f"  {class_name:25} |{bar}| {accuracy*100:.1f}%"


def format_top_intents(top_intents: List[Dict]) -> str:
    """Format top intent predictions as readable string."""
    lines = []
    for i, intent in enumerate(top_intents, 1):
        conf = intent.get("confidence", 0) * 100
        label = intent.get("display_label", "Unknown")
        lines.append(f"  {i}. {label:25} {conf:.1f}%")
    return "\n".join(lines)


def format_processing_time(ms: float) -> str:
    """Format processing time with appropriate unit."""
    if ms < 1:
        return f"{ms*1000:.1f}μs"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"
