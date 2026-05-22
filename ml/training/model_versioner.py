"""
Model versioning utility.

Tracks model versions so you always know which model
is in production and can roll back if needed.
"""
import yaml
from pathlib import Path
from datetime import datetime


VERSION_FILE = Path("ml/saved_models/model_versions.yaml")


def register_model_version(
    model_type: str,
    metrics: dict,
    description: str = "",
) -> str:
    """
    Register a new model version with metadata.

    Returns the version string.
    """
    versions = load_versions()
    version_num = len(versions) + 1
    version_str = f"v{version_num}.0"

    entry = {
        "version": version_str,
        "model_type": model_type,
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": round(float(metrics.get("accuracy", 0)), 4),
        "f1_weighted": round(float(metrics.get("f1_weighted", 0)), 4),
        "description": description,
        "status": "staging",
    }

    versions.append(entry)
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(VERSION_FILE, "w") as f:
        yaml.dump(versions, f, default_flow_style=False)

    return version_str


def promote_to_production(version: str) -> None:
    """Mark a model version as production."""
    versions = load_versions()
    for v in versions:
        if v["version"] == version:
            v["status"] = "production"
        elif v["status"] == "production":
            v["status"] = "archived"

    with open(VERSION_FILE, "w") as f:
        yaml.dump(versions, f, default_flow_style=False)


def load_versions() -> list:
    if not VERSION_FILE.exists():
        return []
    with open(VERSION_FILE) as f:
        return yaml.safe_load(f) or []


def get_production_version() -> dict:
    versions = load_versions()
    for v in versions:
        if v.get("status") == "production":
            return v
    return {}
