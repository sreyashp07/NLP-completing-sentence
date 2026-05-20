"""
Integration tests for FastAPI endpoints.
Requires the API to be running on localhost:8000.

Run:
    py -m pytest tests/integration/ -v
"""
import pytest
import httpx


BASE_URL = "http://localhost:8000/api/v1"


def is_api_running() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not is_api_running(), reason="API not running")
class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = httpx.get(f"{BASE_URL}/health")
        assert r.status_code == 200

    def test_health_response_structure(self):
        r = httpx.get(f"{BASE_URL}/health")
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_model_loaded(self):
        r = httpx.get(f"{BASE_URL}/health")
        assert r.json()["model_loaded"] is True


@pytest.mark.skipif(not is_api_running(), reason="API not running")
class TestPredictEndpoint:
    def test_predict_returns_200(self):
        r = httpx.post(
            f"{BASE_URL}/predict",
            json={"text": "My payment failed but money got deducted"},
        )
        assert r.status_code == 200

    def test_predict_payment_intent(self):
        r = httpx.post(
            f"{BASE_URL}/predict",
            json={"text": "My payment failed but money got deducted"},
        )
        data = r.json()
        assert data["primary_intent"]["intent"] == "payment_issue"

    def test_predict_confidence_range(self):
        r = httpx.post(
            f"{BASE_URL}/predict",
            json={"text": "My payment failed"},
        )
        conf = r.json()["primary_intent"]["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_predict_has_keywords(self):
        r = httpx.post(
            f"{BASE_URL}/predict",
            json={"text": "My payment failed but money was deducted"},
        )
        assert len(r.json()["keywords"]) > 0

    def test_predict_short_text_returns_422(self):
        r = httpx.post(
            f"{BASE_URL}/predict",
            json={"text": "Hi"},
        )
        assert r.status_code == 422

    def test_batch_predict(self):
        r = httpx.post(
            f"{BASE_URL}/predict/batch",
            json={"texts": [
                "My payment failed",
                "I want to cancel my subscription",
                "App keeps crashing",
            ]},
        )
        assert r.status_code == 200
        assert r.json()["total"] == 3
