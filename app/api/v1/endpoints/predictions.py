"""
Prediction API endpoints.

Routes:
  POST /predict       — single text prediction
  POST /predict/batch — batch prediction
"""
import time
from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger

from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter()


def get_model_service(request: Request):
    service = getattr(request.app.state, "model_service", None)
    if not service or not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded. Please retry in a moment.",
        )
    return service


@router.post("/predict", response_model=PredictionResponse)
async def predict_intent(
    payload: PredictionRequest,
    service=Depends(get_model_service),
) -> PredictionResponse:
    """
    Classify intent from a customer support message.

    - **text**: Customer message (3-512 characters)
    - **model_type**: 'baseline' (default) or 'transformer'
    """
    try:
        logger.info(f"Prediction request | text='{payload.text[:80]}...' | model={payload.model_type}")
        result = service.predict(payload.text, payload.model_type)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    payload: BatchPredictionRequest,
    service=Depends(get_model_service),
) -> BatchPredictionResponse:
    """Classify a batch of customer messages (up to 100)."""
    start = time.time()
    try:
        predictions = service.predict_batch(payload.texts)
        elapsed_ms = (time.time() - start) * 1000
        return BatchPredictionResponse(
            success=True,
            total=len(predictions),
            predictions=predictions,
            processing_time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
