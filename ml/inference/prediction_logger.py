"""
Prediction history logger.

Saves prediction results to a local JSONL file
for offline analysis and model monitoring.

In production this would write to a database or
data warehouse like BigQuery or Snowflake.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict


LOG_FILE = Path("logs/predictions.jsonl")


def log_prediction(
    text: str,
    prediction: Dict,
    session_id: str = "default",
) -> None:
    """
    Append a prediction to the JSONL log file.

    Args:
        text: Original input text
        prediction: Full prediction result dict
        session_id: Session identifier for grouping
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    primary = prediction.get("primary_intent", {})
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "text_length": len(text),
        "intent": primary.get("intent", "unknown"),
        "confidence": primary.get("confidence", 0.0),
        "department": primary.get("department", "unknown"),
        "priority": primary.get("priority", "unknown"),
        "processing_time_ms": prediction.get("processing_time_ms", 0),
        "model_used": prediction.get("model_used", "unknown"),
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_prediction_logs(last_n: int = 100) -> list:
    """Read the last N prediction log entries."""
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    records = []
    for line in lines[-last_n:]:
        try:
            records.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue
    return records


def get_log_stats() -> Dict:
    """Get statistics from the prediction log."""
    records = read_prediction_logs(last_n=1000)
    if not records:
        return {"total_logged": 0}

    intents = [r["intent"] for r in records]
    intent_counts: Dict = {}
    for intent in intents:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    avg_conf = sum(r["confidence"] for r in records) / len(records)

    return {
        "total_logged": len(records),
        "avg_confidence": round(avg_conf, 4),
        "intent_distribution": intent_counts,
        "most_common_intent": max(intent_counts, key=intent_counts.get),
    }
