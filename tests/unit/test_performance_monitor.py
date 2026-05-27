"""Unit tests for performance monitor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.evaluation.performance_monitor import PerformanceMonitor

def test_no_data_returns_status():
    monitor = PerformanceMonitor()
    stats = monitor.get_stats()
    assert stats["status"] == "no_data"

def test_record_prediction_updates_count():
    monitor = PerformanceMonitor()
    monitor.record_prediction(12.5)
    monitor.record_prediction(15.0)
    stats = monitor.get_stats()
    assert stats["total_predictions"] == 2

def test_avg_latency_calculated():
    monitor = PerformanceMonitor()
    monitor.record_prediction(10.0)
    monitor.record_prediction(20.0)
    stats = monitor.get_stats()
    assert stats["avg_latency_ms"] == 15.0

def test_error_rate_calculated():
    monitor = PerformanceMonitor()
    monitor.record_prediction(10.0, success=True)
    monitor.record_prediction(10.0, success=False)
    stats = monitor.get_stats()
    assert stats["error_rate"] == 0.5

def test_is_healthy_with_good_metrics():
    monitor = PerformanceMonitor()
    for _ in range(10):
        monitor.record_prediction(50.0, success=True)
    assert monitor.is_healthy() is True

def test_is_unhealthy_with_high_latency():
    monitor = PerformanceMonitor()
    for _ in range(10):
        monitor.record_prediction(1000.0)
    assert monitor.is_healthy(max_avg_latency_ms=100) is False

def test_reset_clears_metrics():
    monitor = PerformanceMonitor()
    monitor.record_prediction(10.0)
    monitor.reset()
    stats = monitor.get_stats()
    assert stats["status"] == "no_data"
