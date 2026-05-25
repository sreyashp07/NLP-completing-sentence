"""
Confusion matrix analyzer for detailed error pattern detection.

Goes beyond basic confusion matrix to identify:
- Most confused intent pairs
- Systematic misclassification patterns
- Per-class error rates
"""
import numpy as np
from typing import Dict, List, Tuple


def get_confused_pairs(
    cm: np.ndarray,
    class_names: list,
    top_n: int = 5,
) -> List[Dict]:
    """
    Find the most commonly confused intent pairs.

    Args:
        cm: Confusion matrix array
        class_names: List of class label strings
        top_n: Number of top confused pairs to return

    Returns:
        List of confused pair dicts sorted by frequency
    """
    pairs = []
    n = len(class_names)

    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append({
                    "true_intent": class_names[i],
                    "predicted_as": class_names[j],
                    "count": int(cm[i][j]),
                    "true_total": int(cm[i].sum()),
                    "error_rate": round(cm[i][j] / cm[i].sum(), 3),
                })

    return sorted(pairs, key=lambda x: x["count"], reverse=True)[:top_n]


def get_per_class_error_rates(
    cm: np.ndarray,
    class_names: list,
) -> List[Dict]:
    """Calculate error rate for each intent class."""
    results = []
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i][i]
        error_rate = 1 - (correct / total) if total > 0 else 0
        results.append({
            "intent": name,
            "total_samples": int(total),
            "correct": int(correct),
            "errors": int(total - correct),
            "error_rate": round(float(error_rate), 4),
            "accuracy": round(1 - float(error_rate), 4),
        })

    return sorted(results, key=lambda x: x["error_rate"], reverse=True)


def get_hardest_intents(
    cm: np.ndarray,
    class_names: list,
    top_n: int = 3,
) -> List[str]:
    """Return the intent classes with highest error rates."""
    error_rates = get_per_class_error_rates(cm, class_names)
    return [r["intent"] for r in error_rates[:top_n] if r["error_rate"] > 0]
