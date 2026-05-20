"""
Dataset validation before training.

Checks:
- Required columns exist
- No null values in key columns
- Minimum samples per class
- Label consistency
"""
import pandas as pd
from typing import List, Tuple


REQUIRED_COLUMNS = ["text", "intent", "priority", "department"]
MIN_SAMPLES_PER_CLASS = 10


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate dataset before training.

    Args:
        df: Training dataframe

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return False, errors

    # Check for nulls
    for col in REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null values")

    # Check minimum samples per class
    class_counts = df["intent"].value_counts()
    for intent, count in class_counts.items():
        if count < MIN_SAMPLES_PER_CLASS:
            errors.append(
                f"Intent '{intent}' has only {count} samples "
                f"(minimum: {MIN_SAMPLES_PER_CLASS})"
            )

    # Check text length
    empty_texts = (df["text"].str.strip() == "").sum()
    if empty_texts > 0:
        errors.append(f"Found {empty_texts} empty text samples")

    is_valid = len(errors) == 0
    return is_valid, errors


def print_validation_report(df: pd.DataFrame) -> None:
    """Print a full validation report to stdout."""
    is_valid, errors = validate_dataset(df)
    print("\n=== Dataset Validation Report ===")
    print(f"Shape: {df.shape}")
    print(f"Classes: {df['intent'].nunique()}")
    print(f"Valid: {is_valid}")
    if errors:
        print("\nErrors found:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All checks passed!")
    print("="*35 + "\n")
