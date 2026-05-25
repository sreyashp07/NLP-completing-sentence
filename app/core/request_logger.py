"""
Structured request logger for API monitoring.

Logs every prediction request with:
- Input text length
- Predicted intent and confidence
- Processing time
- Client information

Enables monitoring dashboards and alerting in production.
"""
from datetime import datetime
from typing import Dict, Optional
from loguru import logger


def log_prediction_request(
    text: str,
    prediction: Dict,
    processing_time_ms: float,
    client_ip: Optional[str] = "unknown",
) -> None:
    """Log a prediction request with full context."""
    primary = prediction.get("primary_intent", {})
    logger.info(
        "PREDICTION | "
        f"ip={client_ip} | "
        f"text_length={len(text)} | "
        f"intent={primary.get('intent', 'unknown')} | "
        f"confidence={primary.get('confidence', 0):.3f} | "
        f"priority={primary.get('priority', 'unknown')} | "
        f"dept={primary.get('department', 'unknown')} | "
        f"time_ms={processing_time_ms:.1f}"
    )


def log_batch_request(
    batch_size: int,
    processing_time_ms: float,
    client_ip: Optional[str] = "unknown",
) -> None:
    """Log a batch prediction request."""
    logger.info(
        "BATCH_PREDICTION | "
        f"ip={client_ip} | "
        f"batch_size={batch_size} | "
        f"time_ms={processing_time_ms:.1f} | "
        f"avg_time_ms={processing_time_ms/max(batch_size,1):.1f}"
    )


def log_model_load(model_type: str, load_time_ms: float) -> None:
    """Log model loading event."""
    logger.info(
        "MODEL_LOAD | "
        f"model={model_type} | "
        f"load_time_ms={load_time_ms:.1f} | "
        f"timestamp={datetime.utcnow().isoformat()}"
    )


def log_error(error: str, context: Optional[Dict] = None) -> None:
    """Log an error with context."""
    logger.error(
        f"ERROR | error={error} | "
        f"context={context or {}} | "
        f"timestamp={datetime.utcnow().isoformat()}"
    )
