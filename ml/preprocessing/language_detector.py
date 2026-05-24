"""
Language detection utility for customer support messages.

In real support systems, customers write in multiple languages.
This module detects non-English messages so they can be
routed to multilingual support agents.

Currently supports: English detection with confidence scoring.
Phase 2: Full multilingual support with langdetect library.
"""
import re
from typing import Dict, Tuple


ENGLISH_COMMON_WORDS = {
    "the", "is", "are", "was", "were", "my", "your", "i", "we",
    "you", "he", "she", "it", "they", "have", "has", "had",
    "not", "but", "and", "or", "for", "with", "from", "this",
    "that", "what", "how", "when", "where", "please", "help",
    "need", "want", "can", "cannot", "will", "would", "should",
    "payment", "account", "order", "refund", "cancel", "issue",
}

NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]")


def detect_language(text: str) -> Dict:
    """
    Detect if text is likely English or another language.

    Args:
        text: Input customer message

    Returns:
        Dict with language, confidence, and is_english flag
    """
    if not text or len(text.strip()) < 3:
        return {"language": "unknown", "confidence": 0.0, "is_english": False}

    words = text.lower().split()
    total_words = len(words)

    if total_words == 0:
        return {"language": "unknown", "confidence": 0.0, "is_english": False}

    # Check English common words
    english_word_count = sum(1 for w in words if w in ENGLISH_COMMON_WORDS)
    english_ratio = english_word_count / total_words

    # Check non-ASCII characters
    non_ascii = len(NON_ASCII_PATTERN.findall(text))
    non_ascii_ratio = non_ascii / len(text)

    # Score
    if non_ascii_ratio > 0.3:
        return {
            "language": "non-english",
            "confidence": round(non_ascii_ratio, 3),
            "is_english": False,
            "non_ascii_ratio": non_ascii_ratio,
        }

    is_english = english_ratio > 0.1 or non_ascii_ratio < 0.05
    confidence = min(0.95, english_ratio + (1 - non_ascii_ratio) * 0.5)

    return {
        "language": "english" if is_english else "non-english",
        "confidence": round(confidence, 3),
        "is_english": is_english,
        "non_ascii_ratio": round(non_ascii_ratio, 3),
    }


def filter_english_texts(texts: list) -> Tuple[list, list]:
    """
    Split texts into English and non-English groups.

    Returns:
        Tuple of (english_texts, non_english_texts)
    """
    english = []
    non_english = []
    for text in texts:
        result = detect_language(text)
        if result["is_english"]:
            english.append(text)
        else:
            non_english.append(text)
    return english, non_english
