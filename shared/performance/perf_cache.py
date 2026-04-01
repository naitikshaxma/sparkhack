from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable


class LruTtlCache:
    def __init__(self, maxsize: int = 512, ttl_seconds: float = 120.0) -> None:
        self.maxsize = max(8, int(maxsize))
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def _expired(self, inserted_at: float) -> bool:
        return (time.time() - inserted_at) > self.ttl_seconds

    def _evict_expired_locked(self) -> None:
        expired_keys = [key for key, (created, _) in self._store.items() if self._expired(created)]
        for key in expired_keys:
            self._store.pop(key, None)

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            created, value = item
            if self._expired(created):
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._evict_expired_locked()
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def get_or_set(self, key: str, builder: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = builder()
        self.set(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            self._evict_expired_locked()
            return len(self._store)


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
