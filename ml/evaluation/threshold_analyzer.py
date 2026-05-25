"""
Confidence threshold analyzer.

Finds the optimal confidence threshold that balances
auto-routing accuracy vs human review rate.

In production you want:
- High confidence predictions auto-routed (saves agent time)
- Low confidence predictions sent to human review (prevents errors)
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


MODEL_DIR = Path("ml/saved_models/baseline")


def analyze_thresholds(
    thresholds: List[float] = None,
) -> List[Dict]:
    """
    Analyze model performance at different confidence thresholds.

    Returns metrics for each threshold showing the tradeoff
    between auto-routing rate and accuracy.
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    try:
        with open(MODEL_DIR / "pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
    except FileNotFoundError:
        return [{"error": "Model not found. Run train_baseline.py first."}]

    import pandas as pd
    from ml.preprocessing.text_cleaner import TextCleaner

    df = pd.read_csv("data/raw/customer_support_dataset.csv")
    cleaner = TextCleaner(remove_stopwords=False, lemmatize=True)
    df["cleaned"] = cleaner.clean_batch(df["text"].tolist())
    df["label"] = le.transform(df["intent"])

    X = df["cleaned"].to_numpy()
    y = df["label"].to_numpy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    probs = pipeline.predict_proba(X_test)
    preds = pipeline.predict(X_test)
    max_confs = probs.max(axis=1)

    results = []
    for threshold in thresholds:
        mask = max_confs >= threshold
        auto_routed = mask.sum()
        auto_rate = auto_routed / len(y_test)

        if auto_routed > 0:
            accuracy = accuracy_score(y_test[mask], preds[mask])
        else:
            accuracy = 0.0

        results.append({
            "threshold": threshold,
            "auto_routed": int(auto_routed),
            "human_review": int(len(y_test) - auto_routed),
            "auto_rate": round(float(auto_rate), 4),
            "accuracy_at_threshold": round(float(accuracy), 4),
        })

    return results


def print_threshold_report() -> None:
    results = analyze_thresholds()
    print("\n" + "="*70)
    print("  CONFIDENCE THRESHOLD ANALYSIS")
    print("="*70)
    print(f"  {'Threshold':>10} | {'Auto-Routed':>12} | {'Auto Rate':>10} | {'Accuracy':>10}")
    print("-"*70)
    for r in results:
        if "error" in r:
            print(f"  Error: {r['error']}")
            break
        print(
            f"  {r['threshold']:>10.2f} | "
            f"{r['auto_routed']:>12} | "
            f"{r['auto_rate']:>9.1%} | "
            f"{r['accuracy_at_threshold']:>9.1%}"
        )
    print("="*70 + "\n")


if __name__ == "__main__":
    print_threshold_report()
