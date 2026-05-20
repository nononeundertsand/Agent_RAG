import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0


class TTLCache:
    def __init__(self, max_size: int = 512, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        item = self._items.get(key)
        now = time.time()
        if not item:
            self.stats.misses += 1
            return None
        expires_at, value = item
        if expires_at < now:
            self._items.pop(key, None)
            self.stats.misses += 1
            return None
        self._items.move_to_end(key)
        self.stats.hits += 1
        return value

    def set(self, key: str, value: Any):
        self._items[key] = (time.time() + self.ttl_seconds, value)
        self._items.move_to_end(key)
        self.stats.writes += 1
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)
            self.stats.evictions += 1

    def snapshot(self) -> dict:
        return {
            "size": len(self._items),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "writes": self.stats.writes,
            "evictions": self.stats.evictions,
        }


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


retrieval_cache = TTLCache(max_size=512, ttl_seconds=300)
generation_cache = TTLCache(max_size=256, ttl_seconds=300)
query_expansion_cache = TTLCache(max_size=1024, ttl_seconds=1800)
