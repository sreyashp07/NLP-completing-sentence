"""Text utility functions for the API layer."""
from typing import List


def sanitize_input(text: str) -> str:
    """Remove control characters and normalize whitespace."""
    import re
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_meaningful_text(text: str, min_words: int = 2) -> bool:
    """Check if text has enough meaningful content."""
    words = [w for w in text.split() if len(w) > 1]
    return len(words) >= min_words


def split_into_chunks(text: str, max_length: int = 512) -> List[str]:
    """Split long text into chunks of max_length characters."""
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:max_length])
        text = text[max_length:]
    return chunks
