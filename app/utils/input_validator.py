"""
Input validation utilities for the prediction pipeline.

Validates and sanitizes all incoming text before
passing to the ML model. Prevents edge cases that
could cause silent prediction errors.
"""
import re
from typing import Tuple


MIN_LENGTH = 3
MAX_LENGTH = 512
MIN_WORD_COUNT = 1


def validate_prediction_input(text: str) -> Tuple[bool, str]:
    """
    Validate a prediction input text.

    Args:
        text: Raw input text

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text:
        return False, "Text cannot be empty"

    if not isinstance(text, str):
        return False, "Text must be a string"

    stripped = text.strip()

    if len(stripped) < MIN_LENGTH:
        return False, f"Text too short (minimum {MIN_LENGTH} characters)"

    if len(stripped) > MAX_LENGTH:
        return False, f"Text too long (maximum {MAX_LENGTH} characters)"

    words = [w for w in stripped.split() if len(w) > 0]
    if len(words) < MIN_WORD_COUNT:
        return False, f"Text must contain at least {MIN_WORD_COUNT} word"

    if re.match(r"^[^a-zA-Z]*$", stripped):
        return False, "Text must contain at least some alphabetic characters"

    return True, ""


def validate_batch_input(texts: list) -> Tuple[bool, str]:
    """Validate a batch of prediction inputs."""
    if not texts:
        return False, "Batch cannot be empty"

    if len(texts) > 100:
        return False, "Batch size cannot exceed 100 texts"

    for i, text in enumerate(texts):
        is_valid, error = validate_prediction_input(text)
        if not is_valid:
            return False, f"Text at index {i}: {error}"

    return True, ""


def sanitize_input(text: str) -> str:
    """Clean and sanitize input text."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:MAX_LENGTH]
