"""
Lightweight metrics tracker for comparing model runs.
Saves metrics history to YAML for trend analysis.
"""
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict


HISTORY_FILE = Path("ml/evaluation/metrics_history.yaml")


def save_run_metrics(metrics: Dict, model_name: str) -> None:
    """Append a training run's metrics to the history file."""
    history = load_history()
    run = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "accuracy": round(float(metrics.get("accuracy", 0)), 4),
        "f1_weighted": round(float(metrics.get("f1_weighted", 0)), 4),
        "f1_macro": round(float(metrics.get("f1_macro", 0)), 4),
    }
    history.append(run)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        yaml.dump(history, f, default_flow_style=False)


def load_history() -> list:
    """Load metrics history from YAML."""
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE) as f:
        return yaml.safe_load(f) or []


def get_best_run(metric: str = "f1_weighted") -> Dict:
    """Return the best run by a given metric."""
    history = load_history()
    if not history:
        return {}
    return max(history, key=lambda x: x.get(metric, 0))
