"""Unit tests for prediction cache."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.utils.cache import PredictionCache

def test_cache_miss_returns_none():
    cache = PredictionCache()
    result = cache.get("my payment failed")
    assert result is None

def test_cache_hit_returns_data():
    cache = PredictionCache()
    data = {"intent": "payment_issue", "confidence": 0.9}
    cache.set("my payment failed", data)
    result = cache.get("my payment failed")
    assert result == data

def test_cache_is_case_insensitive():
    cache = PredictionCache()
    data = {"intent": "payment_issue"}
    cache.set("My Payment Failed", data)
    result = cache.get("my payment failed")
    assert result == data

def test_cache_clear():
    cache = PredictionCache()
    cache.set("test", {"intent": "general_inquiry"})
    cache.clear()
    assert cache.get("test") is None

def test_cache_stats():
    cache = PredictionCache(max_size=100)
    cache.set("test", {"intent": "general_inquiry"})
    stats = cache.stats()
    assert stats["size"] == 1
    assert stats["max_size"] == 100

def test_cache_evicts_when_full():
    cache = PredictionCache(max_size=2)
    cache.set("text1", {"intent": "a"})
    cache.set("text2", {"intent": "b"})
    cache.set("text3", {"intent": "c"})
    assert cache.stats()["size"] <= 2
