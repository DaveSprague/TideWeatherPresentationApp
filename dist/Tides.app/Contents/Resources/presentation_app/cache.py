"""
Simple LRU cache with TTL for in-memory use.
Thread-safe and minimal to avoid external deps.
"""
import time
from collections import OrderedDict
from threading import RLock
from typing import Any, Optional


class LRUCacheTTL:
    def __init__(self, max_size: int = 128, ttl_seconds: int = 900):
        self._store: OrderedDict[Any, Any] = OrderedDict()
        self._ttl = ttl_seconds
        self._max = max_size
        self._lock = RLock()

    def _is_expired(self, expires_at: float) -> bool:
        """Check if cache entry has expired."""
        return expires_at < time.time()

    def get(self, key: Any) -> Optional[Any]:
        """Retrieve value from cache if exists and not expired."""
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            value, expires_at = item
            if self._is_expired(expires_at):
                try:
                    del self._store[key]
                except KeyError:
                    pass
                return None
            # mark as recently used
            self._store.move_to_end(key, last=True)
            return value

    def set(self, key: Any, value: Any) -> None:
        """Store value in cache with TTL."""
        with self._lock:
            expires_at = time.time() + self._ttl
            self._store[key] = (value, expires_at)
            self._store.move_to_end(key, last=True)
            # evict if over capacity
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def delete(self, key: Any) -> None:
        """Remove entry from cache if present."""
        with self._lock:
            try:
                del self._store[key]
            except KeyError:
                pass

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
