"""
Text chunker for long customer messages.

Some customers write very long messages.
This module splits them into processable chunks
while preserving sentence boundaries.
"""
import re
from typing import List


def split_by_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_words(text: str, max_words: int = 50) -> List[str]:
    """Split text into chunks of max_words words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_by_chars(text: str, max_chars: int = 512) -> List[str]:
    """Split text into chunks of max_chars characters."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:max_chars])
        text = text[max_chars:]
    return chunks


def smart_chunk(text: str, max_chars: int = 512) -> List[str]:
    """
    Intelligently chunk text preserving sentence boundaries.

    Tries to keep sentences together up to max_chars limit.
    Falls back to character splitting for very long sentences.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = split_by_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = (
                current_chunk + " " + sentence
                if current_chunk else sentence
            )
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(sentence) > max_chars:
                chunks.extend(chunk_by_chars(sentence, max_chars))
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
