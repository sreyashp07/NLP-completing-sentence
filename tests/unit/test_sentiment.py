"""Unit tests for sentiment analysis module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.sentiment_analyzer import analyze_sentiment


def test_frustrated_sentiment():
    result = analyze_sentiment("This is the worst service ever! Terrible!")
    assert result["sentiment"] == "frustrated"


def test_neutral_sentiment():
    result = analyze_sentiment("I need help with my account login")
    assert result["sentiment"] == "neutral"


def test_positive_sentiment():
    result = analyze_sentiment("Thank you so much, great service, happy!")
    assert result["sentiment"] == "positive"


def test_urgent_detection():
    result = analyze_sentiment("urgent help needed asap account blocked")
    assert result["is_urgent"] is True


def test_exclamation_increases_score():
    result1 = analyze_sentiment("my payment failed")
    result2 = analyze_sentiment("my payment failed!!!")
    assert result2["exclamation_count"] > result1["exclamation_count"]


def test_result_has_required_keys():
    result = analyze_sentiment("help me please")
    required = ["sentiment", "is_urgent", "frustrated_score", "urgent_score"]
    for key in required:
        assert key in result


def test_empty_text():
    result = analyze_sentiment("")
    assert result["sentiment"] in ["neutral", "positive", "frustrated"]
