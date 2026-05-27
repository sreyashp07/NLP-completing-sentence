"""Unit tests for text utility functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.utils.text_utils import sanitize_input, is_meaningful_text, split_into_chunks

def test_sanitize_removes_control_chars():
    result = sanitize_input("hello\x00world")
    assert "\x00" not in result

def test_sanitize_normalizes_whitespace():
    result = sanitize_input("hello   world")
    assert result == "hello world"

def test_is_meaningful_true():
    assert is_meaningful_text("payment failed account") is True

def test_is_meaningful_false():
    assert is_meaningful_text("hi") is False

def test_split_short_text():
    chunks = split_into_chunks("hello world", max_length=512)
    assert len(chunks) == 1

def test_split_long_text():
    chunks = split_into_chunks("a" * 1000, max_length=512)
    assert len(chunks) == 2
