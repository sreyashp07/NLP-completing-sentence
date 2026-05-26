"""
Additional text cleaning utilities for edge cases.

Handles special cases not covered by the main pipeline:
- Phone number masking
- Currency normalization  
- Order ID detection
- Date/time normalization
"""
import re
from typing import Dict, Tuple


def mask_phone_numbers(text: str) -> str:
    """Replace phone numbers with PHONE placeholder."""
    pattern = re.compile(
        r"(\+?\d{1,3}[-.\s]?)?"
        r"(\(?\d{3}\)?[-.\s]?)"
        r"\d{3}[-.\s]?\d{4}"
    )
    return pattern.sub("PHONE", text)


def mask_order_ids(text: str) -> str:
    """Replace order/ticket IDs with ORDER_ID placeholder."""
    pattern = re.compile(
        r"\b(ord|order|ticket|ref|txn|inv|tkt)[-#]?\s*\d{4,12}\b",
        re.IGNORECASE,
    )
    return pattern.sub("ORDER_ID", text)


def normalize_currency(text: str) -> str:
    """Normalize currency mentions to AMOUNT."""
    text = re.sub(r"USD\s*[\d,]+\.?\d*", "AMOUNT", text)
    text = re.sub(r"INR\s*[\d,]+\.?\d*", "AMOUNT", text)
    text = re.sub(r"EUR\s*[\d,]+\.?\d*", "AMOUNT", text)
    return text


def normalize_dates(text: str) -> str:
    """Replace specific dates with DATE placeholder."""
    text = re.sub(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "DATE", text
    )
    text = re.sub(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}\b",
        "DATE", text, flags=re.IGNORECASE,
    )
    return text


def apply_all_masks(text: str) -> Tuple[str, Dict]:
    """
    Apply all masking operations and return masked text
    with a report of what was masked.
    """
    original = text
    masks_applied = {}

    masked = mask_phone_numbers(text)
    if masked != text:
        masks_applied["phone"] = True
    text = masked

    masked = mask_order_ids(text)
    if masked != text:
        masks_applied["order_id"] = True
    text = masked

    masked = normalize_currency(text)
    if masked != text:
        masks_applied["currency"] = True
    text = masked

    masked = normalize_dates(text)
    if masked != text:
        masks_applied["date"] = True
    text = masked

    return text, masks_applied
