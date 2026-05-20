"""Unit tests for dataset validation module."""
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.training.data_validator import validate_dataset


def make_df(n=100):
    return pd.DataFrame({
        "text": [f"sample text {i}" for i in range(n)],
        "intent": ["payment_issue"] * (n // 2) + ["refund_request"] * (n // 2),
        "priority": ["high"] * n,
        "department": ["Billing Team"] * n,
    })


def test_valid_dataset_passes():
    df = make_df(100)
    is_valid, errors = validate_dataset(df)
    assert is_valid is True
    assert len(errors) == 0


def test_missing_column_fails():
    df = make_df(100).drop(columns=["intent"])
    is_valid, errors = validate_dataset(df)
    assert is_valid is False
    assert any("intent" in e for e in errors)


def test_null_values_fail():
    df = make_df(100)
    df.loc[0, "text"] = None
    is_valid, errors = validate_dataset(df)
    assert is_valid is False


def test_empty_text_fails():
    df = make_df(100)
    df.loc[0, "text"] = "   "
    is_valid, errors = validate_dataset(df)
    assert is_valid is False
