"""
Advanced cross-validation utilities for model evaluation.

Provides stratified k-fold CV with detailed reporting.
Used before final model training to tune hyperparameters.
"""
import numpy as np
from typing import Dict, List
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


def run_cross_validation(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Run stratified k-fold cross-validation with multiple metrics.

    Args:
        pipeline: Sklearn pipeline to evaluate
        X: Feature array
        y: Label array
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility

    Returns:
        Dict with mean and std for each metric
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
    }

    results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )

    summary = {}
    for metric in scoring.keys():
        test_scores = results[f"test_{metric}"]
        train_scores = results[f"train_{metric}"]
        summary[metric] = {
            "test_mean": round(float(test_scores.mean()), 4),
            "test_std": round(float(test_scores.std()), 4),
            "train_mean": round(float(train_scores.mean()), 4),
            "overfit_gap": round(
                float(train_scores.mean() - test_scores.mean()), 4
            ),
        }

    return summary


def print_cv_report(summary: Dict) -> None:
    """Print a formatted cross-validation report."""
    print("\n" + "="*60)
    print("  CROSS-VALIDATION REPORT")
    print("="*60)
    for metric, scores in summary.items():
        print(
            f"  {metric:20} | "
            f"test: {scores['test_mean']:.4f} "
            f"(+/-{scores['test_std']:.4f}) | "
            f"overfit gap: {scores['overfit_gap']:.4f}"
        )
    print("="*60 + "\n")
