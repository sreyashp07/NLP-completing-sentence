"""
Hyperparameter tuning utilities for baseline model.

Provides grid search configuration and result analysis
for finding optimal TF-IDF and Logistic Regression params.

Run after initial training to try to squeeze more performance.
"""
from typing import Dict, List


TFIDF_SEARCH_SPACE = {
    "tfidf__max_features": [5000, 10000, 15000, 20000],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__sublinear_tf": [True, False],
    "tfidf__min_df": [1, 2, 3],
}

LR_SEARCH_SPACE = {
    "clf__C": [0.1, 1.0, 5.0, 10.0],
    "clf__solver": ["lbfgs", "saga"],
    "clf__class_weight": ["balanced", None],
}

RECOMMENDED_PARAMS = {
    "tfidf__max_features": 15000,
    "tfidf__ngram_range": (1, 3),
    "tfidf__sublinear_tf": True,
    "tfidf__min_df": 2,
    "clf__C": 5.0,
    "clf__solver": "lbfgs",
    "clf__class_weight": "balanced",
}


def get_search_space(include_tfidf: bool = True, include_lr: bool = True) -> Dict:
    """Get combined hyperparameter search space."""
    space = {}
    if include_tfidf:
        space.update(TFIDF_SEARCH_SPACE)
    if include_lr:
        space.update(LR_SEARCH_SPACE)
    return space


def count_combinations(search_space: Dict) -> int:
    """Count total number of hyperparameter combinations."""
    total = 1
    for values in search_space.values():
        total *= len(values)
    return total


def get_recommended_params() -> Dict:
    """Return recommended params based on experiments."""
    return dict(RECOMMENDED_PARAMS)


def print_search_space_summary() -> None:
    space = get_search_space()
    total = count_combinations(space)
    print("\n=== HYPERPARAMETER SEARCH SPACE ===")
    for param, values in space.items():
        print(f"  {param:40}: {values}")
    print(f"\n  Total combinations: {total}")
    print(f"  With 5-fold CV: {total * 5} model fits")
    print("===================================\n")
