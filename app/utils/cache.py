"""
Simple in-memory prediction cache.

Caches recent predictions to avoid recomputing
identical inputs. Improves response time for
repeated messages (common in support systems).
"""
import hashlib
import time
from typing import Dict, Optional


class PredictionCache:
    """
    LRU-style cache for prediction results.
    Default TTL: 5 minutes. Max size: 500 entries.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict] = {}

    def _make_key(self, text: str, model_type: str = "baseline") -> str:
        raw = f"{text.strip().lower()}:{model_type}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, text: str, model_type: str = "baseline") -> Optional[Dict]:
        key = self._make_key(text, model_type)
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["cached_at"] > self.ttl_seconds:
            del self._cache[key]
            return None
        return entry["data"]

    def set(self, text: str, result: Dict, model_type: str = "baseline") -> None:
        if len(self._cache) >= self.max_size:
            oldest = min(self._cache, key=lambda k: self._cache[k]["cached_at"])
            del self._cache[oldest]
        key = self._make_key(text, model_type)
        self._cache[key] = {
            "data": result,
            "cached_at": time.time(),
        }

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> Dict:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }
