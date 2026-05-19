"""
Unit tests for Pydantic schemas.
Validates request/response models behave correctly.
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.schemas.prediction import (
    PredictionRequest,
    IntentResult,
    TopIntent,
    PredictionResponse,
    BatchPredictionRequest,
    HealthResponse,
)


class TestPredictionRequest:
    def test_valid_request(self):
        req = PredictionRequest(text="My payment failed")
        assert req.text == "My payment failed"

    def test_default_model_type(self):
        req = PredictionRequest(text="My payment failed")
        assert req.model_type == "baseline"

    def test_custom_model_type(self):
        req = PredictionRequest(text="My payment failed", model_type="transformer")
        assert req.model_type == "transformer"

    def test_text_stripped(self):
        req = PredictionRequest(text="  My payment failed  ")
        assert req.text == "My payment failed"

    def test_text_too_short_raises(self):
        with pytest.raises(Exception):
            PredictionRequest(text="Hi")

    def test_text_too_long_raises(self):
        with pytest.raises(Exception):
            PredictionRequest(text="x" * 513)

    def test_empty_text_raises(self):
        with pytest.raises(Exception):
            PredictionRequest(text="")


class TestIntentResult:
    def test_valid_intent_result(self):
        result = IntentResult(
            intent="payment_issue",
            confidence=0.87,
            department="Billing Team",
            priority="critical",
            display_label="Payment Issue",
        )
        assert result.intent == "payment_issue"
        assert result.confidence == 0.87

    def test_confidence_above_1_raises(self):
        with pytest.raises(Exception):
            IntentResult(
                intent="payment_issue",
                confidence=1.5,
                department="Billing Team",
                priority="critical",
                display_label="Payment Issue",
            )

    def test_confidence_below_0_raises(self):
        with pytest.raises(Exception):
            IntentResult(
                intent="payment_issue",
                confidence=-0.1,
                department="Billing Team",
                priority="critical",
                display_label="Payment Issue",
            )


class TestBatchPredictionRequest:
    def test_valid_batch(self):
        req = BatchPredictionRequest(
            texts=["My payment failed", "I want a refund"]
        )
        assert len(req.texts) == 2

    def test_empty_batch_raises(self):
        with pytest.raises(Exception):
            BatchPredictionRequest(texts=[])


class TestHealthResponse:
    def test_valid_health_response(self):
        resp = HealthResponse(
            status="healthy",
            app_name="CustomerIntent",
            version="1.0.0",
            model_loaded=True,
            environment="development",
        )
        assert resp.status == "healthy"
        assert resp.model_loaded is True
        assert isinstance(resp.timestamp, datetime)
