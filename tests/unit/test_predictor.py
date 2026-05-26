"""
Unit tests for the inference engine and intent router.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestBaselinePredictor:

    @pytest.fixture(autouse=True)
    def setup(self):
        model_path = Path("ml/saved_models/baseline/pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found.")
        from ml.inference.predictor import BaselinePredictor
        self.predictor = BaselinePredictor()

    def test_predictor_loads(self):
        assert self.predictor.is_loaded is True

    def test_predict_returns_dict(self):
        result = self.predictor.predict("My payment failed")
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self):
        result = self.predictor.predict("My payment failed")
        for key in ["primary_intent", "top_intents", "keywords", "model_used"]:
            assert key in result

    def test_predict_confidence_range(self):
        result = self.predictor.predict("My payment failed")
        conf = result["primary_intent"]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_predict_payment_intent(self):
        result = self.predictor.predict(
            "My payment failed but money got deducted from my account"
        )
        assert result["primary_intent"]["intent"] == "payment_issue"

    def test_predict_refund_intent(self):
        result = self.predictor.predict(
            "I want to request a refund for my last order"
        )
        assert result["primary_intent"]["intent"] == "refund_request"

    def test_predict_account_locked_intent(self):
        result = self.predictor.predict(
            "My account has been locked and I cannot login"
        )
        assert result["primary_intent"]["intent"] == "account_locked"

    def test_top_intents_count(self):
        result = self.predictor.predict("My payment failed")
        assert len(result["top_intents"]) == 3

    def test_top_intents_sorted(self):
        result = self.predictor.predict("My payment failed")
        confs = [t["confidence"] for t in result["top_intents"]]
        assert confs == sorted(confs, reverse=True)

    def test_predict_batch(self):
        texts = [
            "My payment failed",
            "I want to cancel my subscription",
            "App keeps crashing",
        ]
        results = self.predictor.predict_batch(texts)
        assert len(results) == 3


class TestIntentRouter:

    @pytest.fixture(autouse=True)
    def setup(self):
        model_path = Path("ml/saved_models/baseline/pipeline.pkl")
        if not model_path.exists():
            pytest.skip("Trained model not found.")
        from ml.inference.predictor import IntentRouter
        self.router = IntentRouter(model_type="baseline")

    def test_router_is_ready(self):
        assert self.router.is_ready is True

    def test_route_returns_dict(self):
        result = self.router.route("My payment failed")
        assert isinstance(result, dict)

    def test_route_has_department(self):
        result = self.router.route(
            "My payment failed but money was deducted"
        )
        assert result["primary_intent"]["department"] == "Billing Team"

    def test_route_has_priority(self):
        result = self.router.route("My payment failed")
        assert result["primary_intent"]["priority"] in [
            "critical", "high", "medium", "low"
        ]

    def test_route_critical_payment(self):
        result = self.router.route(
            "My payment failed but money got deducted from my account"
        )
        assert result["primary_intent"]["priority"] == "critical"

    def test_route_batch(self):
        texts = ["My payment failed", "I want a refund"]
        results = self.router.route_batch(texts)
        assert len(results) == 2
