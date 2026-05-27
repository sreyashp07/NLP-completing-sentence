"""
Text deduplication utility.

In production support systems, customers often send
the same message multiple times. This module detects
near-duplicate messages to avoid processing them twice.
"""
import hashlib
from typing import List, Set, Tuple


def get_text_hash(text: str) -> str:
    """Generate MD5 hash of normalized text."""
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate_texts(texts: List[str]) -> Tuple[List[str], List[int]]:
    """
    Remove duplicate texts from a list.

    Args:
        texts: List of input texts

    Returns:
        Tuple of (unique_texts, duplicate_indices)
    """
    seen: Set[str] = set()
    unique_texts = []
    duplicate_indices = []

    for i, text in enumerate(texts):
        text_hash = get_text_hash(text)
        if text_hash not in seen:
            seen.add(text_hash)
            unique_texts.append(text)
        else:
            duplicate_indices.append(i)

    return unique_texts, duplicate_indices


def is_duplicate(text: str, seen_hashes: Set[str]) -> bool:
    """Check if text is a duplicate of already seen texts."""
    return get_text_hash(text) in seen_hashes


def get_similarity_ratio(text1: str, text2: str) -> float:
    """
    Calculate simple word overlap similarity between two texts.
    Returns ratio between 0 and 1.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)
