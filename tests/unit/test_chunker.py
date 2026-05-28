"""Unit tests for text chunker module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.preprocessing.text_chunker import (
    split_by_sentences,
    chunk_by_words,
    chunk_by_chars,
    smart_chunk,
)


def test_split_by_sentences_basic():
    text = "My payment failed. Money was deducted. Please help."
    result = split_by_sentences(text)
    assert len(result) == 3


def test_split_by_sentences_single():
    text = "My payment failed"
    result = split_by_sentences(text)
    assert len(result) == 1


def test_chunk_by_words_short_text():
    text = "payment failed help me"
    result = chunk_by_words(text, max_words=10)
    assert len(result) == 1


def test_chunk_by_words_long_text():
    text = " ".join(["word"] * 100)
    result = chunk_by_words(text, max_words=20)
    assert len(result) == 5


def test_chunk_by_chars_short():
    result = chunk_by_chars("hello world", max_chars=512)
    assert len(result) == 1


def test_chunk_by_chars_long():
    result = chunk_by_chars("a" * 1000, max_chars=512)
    assert len(result) == 2


def test_smart_chunk_short_text():
    text = "My payment failed"
    result = smart_chunk(text, max_chars=512)
    assert result == [text]


def test_smart_chunk_long_text():
    text = " ".join(["word"] * 200)
    result = smart_chunk(text, max_chars=100)
    assert len(result) > 1
    assert all(isinstance(c, str) for c in result)
