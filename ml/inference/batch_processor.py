"""
Batch processing optimizer for high-volume prediction requests.

In production, processing 1000 tickets one-by-one is slow.
This module chunks large batches for efficient processing.
"""
from typing import List, Dict, Generator


def chunk_texts(texts: List[str], chunk_size: int = 32) -> Generator:
    """Split large text lists into chunks for batch processing."""
    for i in range(0, len(texts), chunk_size):
        yield texts[i:i + chunk_size]


def process_large_batch(
    texts: List[str],
    predictor,
    chunk_size: int = 32,
) -> List[Dict]:
    """
    Process large batches efficiently by chunking.

    Args:
        texts: List of customer messages
        predictor: Any predictor with predict_batch() method
        chunk_size: Number of texts per chunk

    Returns:
        List of prediction results
    """
    results = []
    chunks = list(chunk_texts(texts, chunk_size))

    for i, chunk in enumerate(chunks):
        chunk_results = predictor.predict_batch(chunk)
        results.extend(chunk_results)

    return results


def get_batch_summary(predictions: List[Dict]) -> Dict:
    """
    Summarize batch prediction results.

    Returns intent distribution and priority breakdown.
    """
    intent_counts: Dict = {}
    priority_counts: Dict = {}

    for pred in predictions:
        intent = pred.get("primary_intent", {}).get("intent", "unknown")
        priority = pred.get("primary_intent", {}).get("priority", "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    return {
        "total": len(predictions),
        "intent_distribution": intent_counts,
        "priority_distribution": priority_counts,
        "critical_count": priority_counts.get("critical", 0),
    }
