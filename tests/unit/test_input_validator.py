"""Unit tests for input validation utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.input_validator import (
    validate_prediction_input,
    validate_batch_input,
    sanitize_input,
)


def test_valid_input_passes():
    valid, msg = validate_prediction_input("My payment failed")
    assert valid is True
    assert msg == ""


def test_empty_input_fails():
    valid, msg = validate_prediction_input("")
    assert valid is False


def test_too_short_fails():
    valid, msg = validate_prediction_input("hi")
    assert valid is False
    assert "short" in msg


def test_too_long_fails():
    valid, msg = validate_prediction_input("a" * 513)
    assert valid is False
    assert "long" in msg


def test_numbers_only_fails():
    valid, msg = validate_prediction_input("12345678")
    assert valid is False


def test_valid_batch_passes():
    valid, msg = validate_batch_input([
        "My payment failed",
        "I want a refund",
    ])
    assert valid is True


def test_empty_batch_fails():
    valid, msg = validate_batch_input([])
    assert valid is False


def test_oversized_batch_fails():
    valid, msg = validate_batch_input(["text"] * 101)
    assert valid is False


def test_sanitize_removes_control_chars():
    result = sanitize_input("hello\x00world")
    assert "\x00" not in result


def test_sanitize_trims_whitespace():
    result = sanitize_input("  hello world  ")
    assert result == "hello world"
