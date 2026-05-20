"""Unit tests for text statistics utilities."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.text_stats import (
    get_text_length_stats,
    get_vocabulary_size,
    get_class_balance,
    is_balanced,
)


def test_get_text_length_stats():
    texts = ["hello world", "hi", "this is a longer sentence"]
    stats = get_text_length_stats(texts)
    assert "mean" in stats
    assert "min" in stats
    assert "max" in stats
    assert stats["min"] == 2
    assert stats["max"] == 25


def test_get_vocabulary_size():
    texts = ["hello world", "hello python", "world python"]
    size = get_vocabulary_size(texts)
    assert size == 3


def test_get_class_balance():
    labels = ["a", "a", "b", "c", "c", "c"]
    balance = get_class_balance(labels)
    assert balance["c"] == 3
    assert balance["a"] == 2
    assert balance["b"] == 1


def test_is_balanced_true():
    labels = ["a"] * 100 + ["b"] * 100 + ["c"] * 100
    assert is_balanced(labels) is True


def test_is_balanced_false():
    labels = ["a"] * 900 + ["b"] * 10
    assert is_balanced(labels) is False
