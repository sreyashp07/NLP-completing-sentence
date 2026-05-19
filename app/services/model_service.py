"""
Model service layer — sits between API routes and ML inference engine.

Responsibilities:
- Model lifecycle (load, unload, switch)
- Request validation
- Response formatting to match Pydantic schemas
- Error handling and fallback logic
"""
import time
from typing import Dict, List, Optional

from loguru import logger

from ml.inference.predictor import IntentRouter
from app.schemas.prediction import (
    IntentResult,
    PredictionResponse,
    TopIntent,
)


class ModelService:
    """
    Service layer for model management and inference orchestration.
    One instance lives for the duration of the application (singleton via app.state).
    """

    def __init__(self):
        self.router: Optional[IntentRouter] = None
        self.active_model: str = "baseline"
        self._load_timestamp: Optional[float] = None

    def load_model(self, model_type: str = "baseline") -> None:
        """Load a model into memory."""
        logger.info(f"Loading model: {model_type}")
        try:
            self.router = IntentRouter(model_type=model_type)
            self.active_model = model_type
            self._load_timestamp = time.time()
            logger.success(f"Model loaded successfully: {model_type}")
        except Exception as e:
            logger.error(f"Failed to load model '{model_type}': {e}")
            raise

    @property
    def is_ready(self) -> bool:
        return self.router is not None and self.router.is_ready

    def predict(self, text: str, model_type: Optional[str] = None) -> PredictionResponse:
        """
        Run a single prediction and return a validated Pydantic response.

        Args:
            text: Customer support message
            model_type: Optional override for model selection

        Returns:
            PredictionResponse schema
        """
        if not self.is_ready:
            raise RuntimeError("No model loaded. Call load_model() first.")

        start = time.time()
        result = self.router.route(text)
        elapsed_ms = (time.time() - start) * 1000

        primary = result["primary_intent"]

        return PredictionResponse(
            success=True,
            text=text,
            primary_intent=IntentResult(
                intent=primary["intent"],
                confidence=primary["confidence"],
                department=primary["department"],
                priority=primary["priority"],
                display_label=primary["display_label"],
            ),
            top_intents=[
                TopIntent(
                    intent=t["intent"],
                    confidence=t["confidence"],
                    display_label=t["display_label"],
                )
                for t in result["top_intents"]
            ],
            keywords=result["keywords"],
            model_used=result["model_used"],
            processing_time_ms=round(elapsed_ms, 2),
        )

    def predict_batch(self, texts: List[str]) -> List[PredictionResponse]:
        """Run batch prediction."""
        return [self.predict(text) for text in texts]
