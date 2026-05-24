"""Unit tests for text normalization module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.preprocessing.text_normalizer import (
    fix_misspellings,
    normalize_numbers,
    remove_repeated_chars,
    normalize_text,
)


def test_fix_misspellings():
    result = fix_misspellings("pls help asap")
    assert "please" in result
    assert "as soon as possible" in result


def test_normalize_card_number():
    result = normalize_numbers("my card 4111111111111111 was charged")
    assert "CARD_NUMBER" in result


def test_normalize_amount():
    result = normalize_numbers("I was charged $49.99")
    assert "AMOUNT" in result


def test_remove_repeated_chars():
    result = remove_repeated_chars("helllllo pleaseeee")
    assert "lll" not in result
    assert "eee" not in result


def test_normalize_text_pipeline():
    result = normalize_text("plz help meeee asap")
    assert isinstance(result, str)
    assert len(result) > 0


def test_fix_abbreviations():
    result = fix_misspellings("my acc is locked")
    assert "account" in result
