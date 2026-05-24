"""
Analytics endpoint for session metrics.

Returns aggregated stats from prediction history
stored in the application session.
"""
from fastapi import APIRouter, Request
from datetime import datetime
from typing import Dict

router = APIRouter()


@router.get("/analytics/summary")
async def get_analytics_summary(request: Request) -> Dict:
    """
    Returns a summary of prediction activity.
    In production this would query a database.
    For now returns API-level stats.
    """
    return {
        "success": True,
        "message": "Analytics endpoint active",
        "endpoints": {
            "predict": "POST /api/v1/predict",
            "batch": "POST /api/v1/predict/batch",
            "health": "GET /api/v1/health",
        },
        "model_info": {
            "type": "TF-IDF + Logistic Regression",
            "classes": 9,
            "version": "v1.0",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
