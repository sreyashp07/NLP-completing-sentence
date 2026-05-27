"""Unit tests for text deduplication module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.preprocessing.text_deduplicator import (
    get_text_hash, deduplicate_texts,
    is_duplicate, get_similarity_ratio,
)

def test_hash_returns_string():
    result = get_text_hash("my payment failed")
    assert isinstance(result, str)
    assert len(result) == 32

def test_same_text_same_hash():
    h1 = get_text_hash("my payment failed")
    h2 = get_text_hash("my payment failed")
    assert h1 == h2

def test_different_text_different_hash():
    h1 = get_text_hash("payment failed")
    h2 = get_text_hash("account locked")
    assert h1 != h2

def test_case_insensitive_hash():
    h1 = get_text_hash("Payment Failed")
    h2 = get_text_hash("payment failed")
    assert h1 == h2

def test_deduplicate_removes_duplicates():
    texts = ["payment failed", "account locked", "payment failed"]
    unique, dupes = deduplicate_texts(texts)
    assert len(unique) == 2
    assert len(dupes) == 1

def test_similarity_identical_texts():
    ratio = get_similarity_ratio("payment failed", "payment failed")
    assert ratio == 1.0

def test_similarity_no_overlap():
    ratio = get_similarity_ratio("payment failed", "account locked")
    assert ratio == 0.0
