"""
Real-time performance monitor for prediction pipeline.

Tracks latency, throughput and error rates.
Provides alerts when performance degrades.
"""
import time
from typing import Dict, List
from collections import deque


class PerformanceMonitor:
    """
    Lightweight performance monitor for the prediction pipeline.
    Keeps a rolling window of recent prediction times.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._latencies: deque = deque(maxlen=window_size)
        self._errors: deque = deque(maxlen=window_size)
        self._total_predictions = 0
        self._total_errors = 0
        self._start_time = time.time()

    def record_prediction(
        self,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record a prediction event."""
        self._latencies.append(latency_ms)
        self._errors.append(0 if success else 1)
        self._total_predictions += 1
        if not success:
            self._total_errors += 1

    def get_stats(self) -> Dict:
        """Get current performance statistics."""
        if not self._latencies:
            return {"status": "no_data"}

        latencies = list(self._latencies)
        uptime_seconds = time.time() - self._start_time

        return {
            "total_predictions": self._total_predictions,
            "total_errors": self._total_errors,
            "error_rate": round(
                self._total_errors / max(self._total_predictions, 1), 4
            ),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "p95_latency_ms": round(
                sorted(latencies)[int(len(latencies) * 0.95)], 2
            ),
            "uptime_seconds": round(uptime_seconds, 1),
            "throughput_per_min": round(
                self._total_predictions / max(uptime_seconds / 60, 1), 2
            ),
        }

    def is_healthy(
        self,
        max_avg_latency_ms: float = 500,
        max_error_rate: float = 0.05,
    ) -> bool:
        """Check if performance is within acceptable bounds."""
        stats = self.get_stats()
        if stats.get("status") == "no_data":
            return True
        return (
            stats["avg_latency_ms"] <= max_avg_latency_ms
            and stats["error_rate"] <= max_error_rate
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self._latencies.clear()
        self._errors.clear()
        self._total_predictions = 0
        self._total_errors = 0
        self._start_time = time.time()
