"""
NLP Preprocessing package.

Exposes all preprocessing utilities for clean imports.
"""
from ml.preprocessing.text_cleaner import TextCleaner, extract_keywords
from ml.preprocessing.text_normalizer import normalize_text, fix_misspellings
from ml.preprocessing.sentiment_analyzer import analyze_sentiment
from ml.preprocessing.language_detector import detect_language
from ml.preprocessing.text_stats import get_text_length_stats, get_class_balance

__all__ = [
    "TextCleaner",
    "extract_keywords",
    "normalize_text",
    "fix_misspellings",
    "analyze_sentiment",
    "detect_language",
    "get_text_length_stats",
    "get_class_balance",
]
