"""Unit tests for application constants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core.constants import ALL_INTENTS, ALL_PRIORITIES, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH


def test_all_intents_count():
    assert len(ALL_INTENTS) == 9


def test_all_priorities_count():
    assert len(ALL_PRIORITIES) == 4


def test_text_length_constraints():
    assert MIN_TEXT_LENGTH < MAX_TEXT_LENGTH
    assert MIN_TEXT_LENGTH == 3
    assert MAX_TEXT_LENGTH == 512


def test_critical_in_priorities():
    assert "critical" in ALL_PRIORITIES


def test_payment_issue_in_intents():
    assert "payment_issue" in ALL_INTENTS
