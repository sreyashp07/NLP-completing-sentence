"""
Simple in-memory rate limiter for API endpoints.
"""
import time
from collections import defaultdict
from typing import Dict, Tuple


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, list] = defaultdict(list)

    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        now = time.time()
        window_start = now - self.window_seconds
        self._buckets[client_id] = [
            ts for ts in self._buckets[client_id]
            if ts > window_start
        ]
        current_count = len(self._buckets[client_id])
        if current_count >= self.max_requests:
            return False, 0
        self._buckets[client_id].append(now)
        remaining = self.max_requests - current_count - 1
        return True, remaining

    def reset(self, client_id: str) -> None:
        self._buckets[client_id] = []

    def get_usage(self, client_id: str) -> Dict:
        now = time.time()
        window_start = now - self.window_seconds
        active = [
            ts for ts in self._buckets[client_id]
            if ts > window_start
        ]
        return {
            "client_id": client_id,
            "requests_used": len(active),
            "requests_remaining": max(0, self.max_requests - len(active)),
            "limit": self.max_requests,
            "window_seconds": self.window_seconds,
        }
