"""
Application startup validation checks.

Runs before the API starts to ensure all
required files and configurations are present.
Fails fast with clear error messages.
"""
from pathlib import Path
from loguru import logger
from typing import List, Tuple


REQUIRED_CONFIGS = [
    "configs/app_config.yaml",
    "configs/intent_config.yaml",
    "configs/model_config.yaml",
]

REQUIRED_DIRS = [
    "ml/saved_models/baseline",
    "data/raw",
    "logs",
]


def check_required_files() -> Tuple[bool, List[str]]:
    """Check all required config files exist."""
    missing = []
    for config in REQUIRED_CONFIGS:
        if not Path(config).exists():
            missing.append(config)
    return len(missing) == 0, missing


def check_required_directories() -> Tuple[bool, List[str]]:
    """Check all required directories exist."""
    missing = []
    for directory in REQUIRED_DIRS:
        if not Path(directory).exists():
            missing.append(directory)
    return len(missing) == 0, missing


def check_model_artifacts() -> Tuple[bool, List[str]]:
    """Check trained model artifacts exist."""
    model_dir = Path("ml/saved_models/baseline")
    required = ["pipeline.pkl", "label_encoder.pkl"]
    missing = []
    for artifact in required:
        if not (model_dir / artifact).exists():
            missing.append(str(model_dir / artifact))
    return len(missing) == 0, missing


def run_startup_checks() -> bool:
    """
    Run all startup checks.

    Returns True if all checks pass, False otherwise.
    Logs warnings for missing items but does not crash.
    """
    all_passed = True

    configs_ok, missing_configs = check_required_files()
    if not configs_ok:
        logger.warning(f"Missing config files: {missing_configs}")

    dirs_ok, missing_dirs = check_required_directories()
    if not dirs_ok:
        logger.warning(f"Missing directories: {missing_dirs}")
        for d in missing_dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {d}")

    model_ok, missing_artifacts = check_model_artifacts()
    if not model_ok:
        logger.error(
            f"Model artifacts missing: {missing_artifacts}. "
            "Run: python ml/training/train_baseline.py"
        )
        all_passed = False

    if all_passed:
        logger.success("All startup checks passed")
    else:
        logger.warning("Some startup checks failed — check logs above")

    return all_passed
