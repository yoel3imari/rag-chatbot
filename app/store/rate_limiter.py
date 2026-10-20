"""In-memory sliding-window rate limiting."""

import threading
import time
from typing import Dict, List, Tuple


class RateLimiter:
    """Sliding-window rate limiter per client key with periodic stale-bucket sweeping."""

    def __init__(self, max_requests_per_minute: int = 90):
        self.max_rpm = max_requests_per_minute
        self._lock = threading.Lock()
        self._buckets: Dict[str, List[float]] = {}
        self._last_sweep: float = time.time()

    def _sweep_stale_unlocked(self, window_start: float, now: float) -> None:
        if now - self._last_sweep < 60.0 and len(self._buckets) < 500:
            return
        stale_keys = []
        for key, timestamps in self._buckets.items():
            kept = [ts for ts in timestamps if ts > window_start]
            if kept:
                self._buckets[key] = kept
            else:
                stale_keys.append(key)
        for key in stale_keys:
            self._buckets.pop(key, None)
        self._last_sweep = now

    def check(self, client_key: str) -> Tuple[bool, int]:
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            self._sweep_stale_unlocked(window_start, now)
            timestamps = [ts for ts in self._buckets.get(client_key, []) if ts > window_start]
            if len(timestamps) >= self.max_rpm:
                retry_after = max(1, int(60.0 - (now - timestamps[0])))
                self._buckets[client_key] = timestamps
                return False, retry_after
            timestamps.append(now)
            self._buckets[client_key] = timestamps
            return True, 0
