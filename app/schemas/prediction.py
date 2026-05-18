"""
Pydantic schemas for API request and response validation.
These are the contracts between frontend, API, and ML pipeline.
"""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Single text prediction request."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="Customer support message to classify",
        examples=["My payment failed but money got deducted"],
    )
    model_type: Optional[str] = Field(
        default="baseline",
        description="Model to use: 'baseline' or 'transformer'",
    )

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        return v.strip()


class IntentResult(BaseModel):
    """Single intent prediction result."""

    intent: str = Field(..., description="Predicted intent label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    department: str = Field(..., description="Routed department")
    priority: str = Field(..., description="Ticket priority level")
    display_label: str = Field(..., description="Human-readable intent label")


class TopIntent(BaseModel):
    """Top-N intent candidates with scores."""

    intent: str
    confidence: float
    display_label: str


class PredictionResponse(BaseModel):
    """Full API response for a prediction request."""

    success: bool = True
    text: str
    primary_intent: IntentResult
    top_intents: List[TopIntent] = Field(
        description="Top 3 intent candidates ranked by confidence"
    )
    keywords: List[str] = Field(description="Important keywords detected")
    model_used: str
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionRequest(BaseModel):
    """Batch prediction for multiple texts."""

    texts: List[str] = Field(..., min_length=1, max_length=100)
    model_type: Optional[str] = "baseline"


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    success: bool = True
    total: int
    predictions: List[PredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    app_name: str
    version: str
    model_loaded: bool
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
