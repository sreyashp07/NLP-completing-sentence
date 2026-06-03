"""Unit tests for color constants module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.core.colors import (
    PRIORITY_COLORS, INTENT_PALETTE,
    BG_PRIMARY, TEXT_PRIMARY, ACCENT_PRIMARY,
)


def test_priority_colors_has_all_keys():
    for key in ["critical", "high", "medium", "low"]:
        assert key in PRIORITY_COLORS


def test_priority_colors_are_hex():
    for color in PRIORITY_COLORS.values():
        assert color.startswith("#")
        assert len(color) == 7


def test_intent_palette_has_9_colors():
    assert len(INTENT_PALETTE) == 9


def test_all_palette_colors_are_hex():
    for color in INTENT_PALETTE:
        assert color.startswith("#")
        assert len(color) == 7


def test_critical_is_red_tone():
    assert PRIORITY_COLORS["critical"] == "#E63946"


def test_background_colors_defined():
    assert BG_PRIMARY == "#0A0E1A"


def test_accent_primary_is_blue():
    assert ACCENT_PRIMARY == "#4086F5"
