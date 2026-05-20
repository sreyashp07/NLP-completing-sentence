"""
Text statistics utilities for dataset analysis.
Provides length stats, vocabulary size, and class balance metrics.
"""
from typing import Dict, List
import numpy as np


def get_text_length_stats(texts: List[str]) -> Dict:
    lengths = [len(t) for t in texts]
    return {
        "mean": round(float(np.mean(lengths)), 2),
        "median": round(float(np.median(lengths)), 2),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "std": round(float(np.std(lengths)), 2),
    }


def get_vocabulary_size(texts: List[str]) -> int:
    words = set()
    for text in texts:
        words.update(text.lower().split())
    return len(words)


def get_class_balance(labels: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def is_balanced(labels: List[str], threshold: float = 0.1) -> bool:
    counts = list(get_class_balance(labels).values())
    if not counts:
        return True
    ratio = (max(counts) - min(counts)) / max(counts)
    return ratio <= threshold
