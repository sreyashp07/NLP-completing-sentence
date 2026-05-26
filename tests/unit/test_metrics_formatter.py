"""Unit tests for metrics formatting utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.metrics_formatter import (
    format_metrics_for_display,
    format_confusion_matrix_row,
    format_top_intents,
    format_processing_time,
)


def test_format_metrics_converts_floats():
    metrics = {"accuracy": 0.95, "f1_weighted": 0.94}
    result = format_metrics_for_display(metrics)
    assert result["accuracy"] == "95.0%"
    assert result["f1_weighted"] == "94.0%"


def test_format_confusion_matrix_row():
    result = format_confusion_matrix_row("payment_issue", 38, 40)
    assert "payment_issue" in result
    assert "95.0%" in result


def test_format_top_intents():
    intents = [
        {"display_label": "Payment Issue", "confidence": 0.91},
        {"display_label": "Refund Request", "confidence": 0.05},
    ]
    result = format_top_intents(intents)
    assert "Payment Issue" in result
    assert "91.0%" in result


def test_format_processing_time_ms():
    assert "ms" in format_processing_time(12.5)


def test_format_processing_time_seconds():
    assert "s" in format_processing_time(1500.0)


def test_format_processing_time_microseconds():
    assert "μs" in format_processing_time(0.5)
