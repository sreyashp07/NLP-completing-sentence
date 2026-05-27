"""Unit tests for rate limiter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.core.rate_limiter import RateLimiter

def test_allows_request_within_limit():
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    allowed, remaining = limiter.is_allowed("client1")
    assert allowed is True
    assert remaining == 4

def test_blocks_after_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.is_allowed("client1")
    allowed, remaining = limiter.is_allowed("client1")
    assert allowed is False
    assert remaining == 0

def test_reset_clears_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.is_allowed("client1")
    limiter.is_allowed("client1")
    limiter.reset("client1")
    allowed, _ = limiter.is_allowed("client1")
    assert allowed is True

def test_different_clients_independent():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.is_allowed("client1")
    allowed, _ = limiter.is_allowed("client2")
    assert allowed is True

def test_get_usage_returns_stats():
    limiter = RateLimiter(max_requests=10, window_seconds=60)
    limiter.is_allowed("client1")
    usage = limiter.get_usage("client1")
    assert usage["requests_used"] == 1
    assert usage["requests_remaining"] == 9
    assert usage["limit"] == 10
