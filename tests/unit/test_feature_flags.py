"""Unit tests for feature flags system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.core.feature_flags import (
    is_enabled, enable, disable,
    get_all_flags, get_enabled_features,
)

def test_enabled_flag_returns_true():
    assert is_enabled("live_typing_prediction") is True

def test_disabled_flag_returns_false():
    assert is_enabled("transformer_model") is False

def test_unknown_flag_returns_false():
    assert is_enabled("nonexistent_feature") is False

def test_enable_flag():
    disable("rate_limiting")
    enable("rate_limiting")
    assert is_enabled("rate_limiting") is True
    disable("rate_limiting")

def test_disable_flag():
    enable("rate_limiting")
    disable("rate_limiting")
    assert is_enabled("rate_limiting") is False

def test_get_all_flags_returns_dict():
    flags = get_all_flags()
    assert isinstance(flags, dict)
    assert len(flags) > 0

def test_get_enabled_features_returns_list():
    features = get_enabled_features()
    assert isinstance(features, list)
    assert "live_typing_prediction" in features
