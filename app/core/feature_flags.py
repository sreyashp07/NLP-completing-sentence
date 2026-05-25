"""
Feature flags for gradual feature rollout.

Allows enabling/disabling features without redeployment.
In production, connect to LaunchDarkly or AWS AppConfig.
For now, uses a simple config-driven approach.
"""
from typing import Dict


# Feature flag registry
# Set to True to enable, False to disable
FLAGS: Dict[str, bool] = {
    "live_typing_prediction": True,
    "batch_prediction": True,
    "confidence_calibration": True,
    "sentiment_analysis": True,
    "advanced_routing": True,
    "prediction_cache": True,
    "rate_limiting": False,
    "analytics_endpoint": True,
    "multilingual_support": False,
    "transformer_model": False,
    "shap_explainability": False,
    "webhook_notifications": False,
}


def is_enabled(flag_name: str) -> bool:
    """Check if a feature flag is enabled."""
    return FLAGS.get(flag_name, False)


def enable(flag_name: str) -> None:
    """Enable a feature flag at runtime."""
    if flag_name in FLAGS:
        FLAGS[flag_name] = True


def disable(flag_name: str) -> None:
    """Disable a feature flag at runtime."""
    if flag_name in FLAGS:
        FLAGS[flag_name] = False


def get_all_flags() -> Dict[str, bool]:
    """Return all feature flags and their current state."""
    return dict(FLAGS)


def get_enabled_features() -> list:
    """Return list of currently enabled feature names."""
    return [k for k, v in FLAGS.items() if v]
