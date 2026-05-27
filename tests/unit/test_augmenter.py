"""Unit tests for text augmentation module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.preprocessing.augmenter import (
    add_opener, add_closer, dropout_words,
    augment_text, augment_dataset,
)

def test_add_opener_returns_string():
    result = add_opener("my payment failed")
    assert isinstance(result, str)
    assert len(result) > 0

def test_add_closer_returns_string():
    result = add_closer("my payment failed")
    assert isinstance(result, str)

def test_dropout_keeps_short_text():
    text = "hi help me"
    result = dropout_words(text, dropout_rate=0.9)
    assert isinstance(result, str)
    assert len(result) > 0

def test_augment_text_returns_string():
    result = augment_text("my payment failed")
    assert isinstance(result, str)
    assert len(result) > 0

def test_augment_dataset_multiplier():
    texts = ["payment failed", "account locked", "need refund"]
    augmented = augment_dataset(texts, multiplier=3)
    assert len(augmented) == 9

def test_augment_dataset_returns_strings():
    texts = ["payment failed"]
    augmented = augment_dataset(texts, multiplier=2)
    assert all(isinstance(t, str) for t in augmented)
