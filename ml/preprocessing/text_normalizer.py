"""
Text normalization utilities for customer support messages.

Handles number normalization, repeated character removal,
and common misspelling corrections seen in support tickets.
"""
import re
from typing import Dict


COMMON_MISSPELLINGS: Dict[str, str] = {
    "recieve": "receive",
    "occured": "occurred",
    "transfered": "transferred",
    "recieved": "received",
    "cant": "cannot",
    "wont": "will not",
    "didnt": "did not",
    "doesnt": "does not",
    "havent": "have not",
    "wasnt": "was not",
    "werent": "were not",
    "ive": "i have",
    "id": "i would",
    "yrs": "years",
    "mins": "minutes",
    "msg": "message",
    "acc": "account",
    "txn": "transaction",
    "amt": "amount",
    "ref": "reference",
    "pls": "please",
    "plz": "please",
    "asap": "as soon as possible",
    "ur": "your",
    "u": "you",
    "r": "are",
    "thx": "thanks",
    "ty": "thank you",
}


def fix_misspellings(text: str) -> str:
    """Correct common misspellings in support messages."""
    words = text.lower().split()
    corrected = [COMMON_MISSPELLINGS.get(w, w) for w in words]
    return " ".join(corrected)


def normalize_numbers(text: str) -> str:
    """Replace specific numbers with generic placeholders."""
    text = re.sub(r"\b\d{10,16}\b", "CARD_NUMBER", text)
    text = re.sub(r"\b\d{6,9}\b", "ORDER_ID", text)
    text = re.sub(r"\$[\d,]+\.?\d*", "AMOUNT", text)
    text = re.sub(r"₹[\d,]+\.?\d*", "AMOUNT", text)
    return text


def remove_repeated_chars(text: str) -> str:
    """Fix repeated characters like 'hellooooo' -> 'hello'."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_text(text: str) -> str:
    """Apply full normalization pipeline."""
    text = remove_repeated_chars(text)
    text = fix_misspellings(text)
    return text
