"""Unit tests for language detection module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.language_detector import detect_language, filter_english_texts


def test_english_text_detected():
    result = detect_language("My payment failed and I need help")
    assert result["is_english"] is True
    assert result["language"] == "english"


def test_empty_text_returns_unknown():
    result = detect_language("")
    assert result["language"] == "unknown"


def test_short_text_returns_unknown():
    result = detect_language("hi")
    assert result["language"] == "unknown"


def test_confidence_is_float():
    result = detect_language("My account is locked please help")
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_filter_english_texts():
    texts = [
        "My payment failed",
        "I need a refund please",
        "App is crashing on my phone",
    ]
    english, non_english = filter_english_texts(texts)
    assert len(english) > 0
    assert len(english) + len(non_english) == len(texts)


def test_result_has_required_keys():
    result = detect_language("My payment failed")
    assert "language" in result
    assert "confidence" in result
    assert "is_english" in result
