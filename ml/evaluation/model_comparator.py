"""
Model comparison utility.

When you train multiple models (baseline vs transformer),
use this to compare their metrics side by side.
"""
import yaml
from pathlib import Path
from typing import Dict, List


def load_model_metrics(model_type: str) -> Dict:
    """Load saved metrics for a model type."""
    path = Path(f"ml/saved_models/{model_type}/metrics.yaml")
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def compare_models(model_types: List[str]) -> Dict:
    """
    Compare metrics across multiple model types.

    Args:
        model_types: List of model type strings

    Returns:
        Comparison dict with winner for each metric
    """
    all_metrics = {}
    for model in model_types:
        metrics = load_model_metrics(model)
        if metrics:
            all_metrics[model] = metrics

    if not all_metrics:
        return {"error": "No model metrics found"}

    comparison = {"models": all_metrics, "winners": {}}

    for metric in ["accuracy", "f1_weighted", "f1_macro"]:
        best_model = max(
            all_metrics,
            key=lambda m: all_metrics[m].get(metric, 0),
        )
        comparison["winners"][metric] = {
            "model": best_model,
            "score": all_metrics[best_model].get(metric, 0),
        }

    return comparison


def print_comparison_report(model_types: List[str]) -> None:
    """Print a formatted model comparison report."""
    comparison = compare_models(model_types)
    if "error" in comparison:
        print(f"Error: {comparison['error']}")
        return

    print("\n" + "="*60)
    print("  MODEL COMPARISON REPORT")
    print("="*60)

    for model, metrics in comparison["models"].items():
        print(f"\n  {model.upper()}")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"    {key:20}: {val:.4f}")

    print("\n  WINNERS")
    for metric, winner in comparison["winners"].items():
        print(f"    {metric:20}: {winner['model']} ({winner['score']:.4f})")
    print("="*60 + "\n")
