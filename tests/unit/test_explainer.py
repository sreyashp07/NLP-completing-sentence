"""Unit tests for intent explainability module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pytest


class TestExplainer:

    @pytest.fixture(autouse=True)
    def setup(self):
        model_path = Path("ml/saved_models/baseline/pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Model not found. Run train_baseline.py first.")
        from ml.inference.intent_explainer import explain_prediction
        self.explain = explain_prediction

    def test_explain_returns_dict(self):
        pred = {
            "primary_intent": {
                "intent": "payment_issue",
                "confidence": 0.91,
            }
        }
        result = self.explain("my payment failed", pred)
        assert isinstance(result, dict)

    def test_explain_has_required_keys(self):
        pred = {
            "primary_intent": {
                "intent": "payment_issue",
                "confidence": 0.91,
            }
        }
        result = self.explain("my payment failed", pred)
        assert "explanation" in result
        assert "confidence_label" in result

    def test_high_confidence_label(self):
        pred = {
            "primary_intent": {
                "intent": "payment_issue",
                "confidence": 0.95,
            }
        }
        result = self.explain("my payment failed", pred)
        assert result.get("confidence_label") == "very high"
