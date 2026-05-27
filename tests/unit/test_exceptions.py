"""Unit tests for custom exceptions."""
import sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.core.exceptions import (
    CustomerIntentError, ModelNotLoadedError,
    ModelLoadError, InvalidTextError, PredictionError,
)

def test_model_not_loaded_error():
    with pytest.raises(ModelNotLoadedError):
        raise ModelNotLoadedError()

def test_model_load_error():
    err = ModelLoadError("ml/saved_models", "file not found")
    assert "ml/saved_models" in str(err)

def test_invalid_text_error():
    err = InvalidTextError("text too short")
    assert "text too short" in str(err)

def test_prediction_error():
    err = PredictionError("pipeline failed")
    assert "pipeline failed" in str(err)

def test_all_inherit_from_base():
    assert issubclass(ModelNotLoadedError, CustomerIntentError)
    assert issubclass(ModelLoadError, CustomerIntentError)
    assert issubclass(InvalidTextError, CustomerIntentError)
    assert issubclass(PredictionError, CustomerIntentError)
