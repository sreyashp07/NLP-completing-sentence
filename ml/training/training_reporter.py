"""
Training progress reporter.

Generates a clean training summary report after model training.
Saves report to ml/evaluation/reports/training_report.txt
"""
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict


REPORT_PATH = Path("ml/evaluation/reports/training_report.txt")


def generate_training_report(
    model_name: str,
    metrics: Dict,
    params: Dict,
    dataset_info: Dict,
) -> str:
    """
    Generate a human-readable training report.

    Args:
        model_name: Name of trained model
        metrics: Evaluation metrics dict
        params: Model hyperparameters
        dataset_info: Dataset statistics

    Returns:
        Formatted report string
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 60

    lines = [
        sep,
        f"  TRAINING REPORT — {model_name}",
        f"  Generated: {now}",
        sep,
        "",
        "  DATASET",
        f"    Total samples   : {dataset_info.get('total_samples', 'N/A')}",
        f"    Training samples: {dataset_info.get('train_samples', 'N/A')}",
        f"    Test samples    : {dataset_info.get('test_samples', 'N/A')}",
        f"    Classes         : {dataset_info.get('num_classes', 'N/A')}",
        "",
        "  METRICS",
        f"    Accuracy        : {metrics.get('accuracy', 0):.4f}",
        f"    F1 Weighted     : {metrics.get('f1_weighted', 0):.4f}",
        f"    F1 Macro        : {metrics.get('f1_macro', 0):.4f}",
        "",
        "  HYPERPARAMETERS",
    ]

    for key, val in params.items():
        lines.append(f"    {key:20}: {val}")

    lines += ["", sep, ""]
    return "\n".join(lines)


def save_training_report(report: str) -> None:
    """Save training report to disk."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)


def quick_report(model_name: str, metrics: Dict) -> None:
    """Print a quick training summary to console."""
    print(f"\n{'='*50}")
    print(f"  {model_name} Training Complete")
    print(f"  Accuracy  : {metrics.get('accuracy', 0):.4f}")
    print(f"  F1 Weighted: {metrics.get('f1_weighted', 0):.4f}")
    print(f"{'='*50}\n")
