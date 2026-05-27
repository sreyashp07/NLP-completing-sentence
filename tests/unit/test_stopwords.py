"""Unit tests for stopwords configuration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.preprocessing.stopwords_config import FINAL_STOPWORDS, KEEP_WORDS

def test_negations_preserved():
    assert "not" not in FINAL_STOPWORDS
    assert "never" not in FINAL_STOPWORDS
    assert "no" not in FINAL_STOPWORDS

def test_keep_words_not_in_stopwords():
    for word in KEEP_WORDS:
        assert word not in FINAL_STOPWORDS

def test_common_words_removed():
    assert "the" in FINAL_STOPWORDS
    assert "a" in FINAL_STOPWORDS
    assert "is" in FINAL_STOPWORDS

def test_domain_stopwords_added():
    assert "hello" in FINAL_STOPWORDS
    assert "hi" in FINAL_STOPWORDS
