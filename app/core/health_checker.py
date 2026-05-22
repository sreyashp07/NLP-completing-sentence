"""
Application health checker.

Checks all critical dependencies are available:
- ML model loaded
- Data directory accessible
- Config files present
"""
from pathlib import Path
from typing import Dict


def check_model_health() -> Dict:
    model_dir = Path("ml/saved_models/baseline")
    pipeline_exists = (model_dir / "pipeline.pkl").exists()
    le_exists = (model_dir / "label_encoder.pkl").exists()
    return {
        "status": "ok" if pipeline_exists and le_exists else "missing",
        "pipeline": pipeline_exists,
        "label_encoder": le_exists,
    }


def check_data_health() -> Dict:
    data_path = Path("data/raw/customer_support_dataset.csv")
    return {
        "status": "ok" if data_path.exists() else "missing",
        "dataset_exists": data_path.exists(),
    }


def check_config_health() -> Dict:
    config_dir = Path("configs")
    required = [
        "app_config.yaml",
        "intent_config.yaml",
        "model_config.yaml",
    ]
    missing = [f for f in required if not (config_dir / f).exists()]
    return {
        "status": "ok" if not missing else "missing",
        "missing_configs": missing,
    }


def full_health_check() -> Dict:
    model = check_model_health()
    data = check_data_health()
    config = check_config_health()

    all_ok = all(
        c["status"] == "ok"
        for c in [model, data, config]
    )

    return {
        "overall": "healthy" if all_ok else "degraded",
        "model": model,
        "data": data,
        "config": config,
    }
