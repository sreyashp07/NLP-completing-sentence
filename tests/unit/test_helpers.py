"""Unit tests for utility helper functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.helpers import (
    format_confidence,
    get_priority_color,
    truncate_text,
    generate_ticket_id,
)


def test_format_confidence():
    assert format_confidence(0.873) == "87.3%"
    assert format_confidence(1.0) == "100.0%"
    assert format_confidence(0.0) == "0.0%"


def test_get_priority_color():
    assert get_priority_color("critical") == "#FF2D55"
    assert get_priority_color("low") == "#34C759"
    assert get_priority_color("unknown") == "#888888"


def test_truncate_text():
    assert truncate_text("hello", 10) == "hello"
    assert truncate_text("a" * 200, 100).endswith("...")
    assert len(truncate_text("a" * 200, 100)) == 103


def test_generate_ticket_id():
    assert generate_ticket_id(0) == "TKT-1001"
    assert generate_ticket_id(9) == "TKT-1010"
