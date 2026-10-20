"""TTL cache used for upload and chat idempotency keys."""

import time
from typing import Any, Optional


class IdempotencyMixin:
    def _sweep_idempotency_unlocked(self, now: float) -> None:
        expired = [k for k, (exp_ts, _) in self._idempotency_cache.items() if now >= exp_ts]
        for key in expired:
            self._idempotency_cache.pop(key, None)

    def check_idempotency(self, key: Optional[str]) -> Optional[Any]:
        if not key:
            return None
        now = time.time()
        with self._lock:
            self._sweep_idempotency_unlocked(now)
            if key in self._idempotency_cache:
                return self._idempotency_cache[key][1]
            return None

    def save_idempotency(self, key: Optional[str], result: Any, ttl_seconds: float = 300.0) -> None:
        if not key:
            return
        now = time.time()
        with self._lock:
            self._sweep_idempotency_unlocked(now)
            self._idempotency_cache[key] = (now + ttl_seconds, result)

    def release_idempotency(self, key: Optional[str]) -> None:
        if not key:
            return
        with self._lock:
            self._idempotency_cache.pop(key, None)
