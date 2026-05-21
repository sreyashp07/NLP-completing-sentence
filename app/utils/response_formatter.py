"""
Response formatting utilities for consistent API output.
Ensures all responses follow the same structure and formatting rules.
"""
from datetime import datetime
from typing import Any, Dict, Optional


def format_success_response(data: Any, message: str = "Success") -> Dict:
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }


def format_error_response(
    error: str,
    code: int = 500,
    details: Optional[str] = None,
) -> Dict:
    return {
        "success": False,
        "error": error,
        "code": code,
        "details": details,
        "timestamp": datetime.utcnow().isoformat(),
    }


def format_prediction_summary(prediction: Dict) -> str:
    p = prediction.get("primary_intent", {})
    return (
        f"Intent: {p.get('display_label')} "
        f"({p.get('confidence', 0)*100:.1f}%) | "
        f"Dept: {p.get('department')} | "
        f"Priority: {p.get('priority', '').upper()}"
    )
