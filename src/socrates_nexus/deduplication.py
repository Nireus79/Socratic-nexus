"""Request deduplication to eliminate duplicate API calls and save costs."""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Cached LLM response."""

    request_hash: str
    response: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 3600  # 1 hour default
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.utcnow() > (self.created_at + timedelta(seconds=self.ttl_seconds))


class RequestDeduplicator:
    """Deduplicates identical LLM requests to save API calls and costs."""

    def __init__(self, max_cache_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize deduplicator.

        Args:
            max_cache_size: Maximum number of cached responses
            default_ttl: Default TTL for cache entries in seconds
        """
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CachedResponse] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cost_saved_usd": 0.0,
        }

    def _hash_request(self, model: str, messages: list, **kwargs) -> str:
        """
        Create hash of request for deduplication.

        Args:
            model: LLM model name
            messages: Message list
            **kwargs: Additional parameters

        Returns:
            Hash string
        """
        # Create canonical representation
        request_dict = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        # Remove non-deterministic fields
        request_dict.pop("timestamp", None)
        request_dict.pop("request_id", None)
        request_dict.pop("trace_id", None)

        # Serialize to JSON
        request_json = json.dumps(request_dict, sort_keys=True, default=str)

        # Hash
        return hashlib.sha256(request_json.encode()).hexdigest()

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _evict_lru(self) -> None:
        """Evict least recently used entries if cache is full."""
        if len(self._cache) >= self.max_cache_size:
            # Sort by hit count and creation time
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1].hit_count, x[1].created_at),
            )

            # Remove oldest 10%
            evict_count = max(1, self.max_cache_size // 10)
            for key, _ in sorted_entries[:evict_count]:
                del self._cache[key]

            logger.debug(f"Evicted {evict_count} cache entries (LRU)")

    async def get_cached_response(
        self,
        model: str,
        messages: list,
        **kwargs,
    ) -> Optional[Any]:
        """
        Get cached response if exists.

        Args:
            model: LLM model
            messages: Message list
            **kwargs: Additional parameters

        Returns:
            Cached response or None
        """
        async with self._lock:
            request_hash = self._hash_request(model, messages, **kwargs)

            # Cleanup expired entries
            await self._cleanup_expired()

            # Check cache
            if request_hash in self._cache:
                entry = self._cache[request_hash]

                if not entry.is_expired():
                    entry.hit_count += 1
                    self._stats["total_requests"] += 1
                    self._stats["cache_hits"] += 1

                    logger.debug(
                        f"Cache hit for {model} (hash: {request_hash[:8]}..., hits: {entry.hit_count})"
                    )
                    return entry.response
                else:
                    # Expired, remove
                    del self._cache[request_hash]

            self._stats["total_requests"] += 1
            self._stats["cache_misses"] += 1
            return None

    async def cache_response(
        self,
        model: str,
        messages: list,
        response: Any,
        cost_usd: float = 0.0,
        ttl_seconds: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Cache LLM response.

        Args:
            model: LLM model
            messages: Message list
            response: LLM response
            cost_usd: Cost of the request in USD
            ttl_seconds: TTL for this entry
            **kwargs: Additional parameters
        """
        async with self._lock:
            request_hash = self._hash_request(model, messages, **kwargs)

            # Cleanup if needed
            await self._evict_lru()

            # Cache response
            entry = CachedResponse(
                request_hash=request_hash,
                response=response,
                ttl_seconds=ttl_seconds or self.default_ttl,
            )

            self._cache[request_hash] = entry
            self._stats["cost_saved_usd"] += cost_usd

            logger.debug(
                f"Cached response for {model} (hash: {request_hash[:8]}..., cost saved: ${cost_usd:.4f})"
            )

    async def clear_cache(self) -> None:
        """Clear all cached responses."""
        async with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    async def clear_old_entries(self, max_age_seconds: int) -> int:
        """
        Clear entries older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of entries removed
        """
        async with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
            old_keys = [key for key, entry in self._cache.items() if entry.created_at < cutoff_time]

            for key in old_keys:
                del self._cache[key]

            logger.info(f"Removed {len(old_keys)} old cache entries")
            return len(old_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        hit_rate = (
            (self._stats["cache_hits"] / self._stats["total_requests"] * 100)
            if self._stats["total_requests"] > 0
            else 0
        )

        return {
            "total_requests": self._stats["total_requests"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "hit_rate_percent": hit_rate,
            "cost_saved_usd": self._stats["cost_saved_usd"],
            "cache_size": len(self._cache),
            "max_cache_size": self.max_cache_size,
        }

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].hit_count,
            reverse=True,
        )

        return {
            "total_entries": len(self._cache),
            "top_cached": [
                {
                    "hash": key[:16],
                    "hits": entry.hit_count,
                    "age_minutes": ((datetime.utcnow() - entry.created_at).total_seconds() / 60),
                    "expired": entry.is_expired(),
                }
                for key, entry in sorted_entries[:10]
            ],
        }


class RequestBatcher:
    """Batches similar requests for efficient processing."""

    def __init__(self, batch_window_ms: int = 100, max_batch_size: int = 10):
        """
        Initialize batcher.

        Args:
            batch_window_ms: Time window for batching in milliseconds
            max_batch_size: Maximum batch size
        """
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self._pending: Dict[str, list] = {}
        self._lock = asyncio.Lock()

    async def add_request(self, batch_key: str, request: Any) -> None:
        """
        Add request to batch.

        Args:
            batch_key: Key for grouping similar requests
            request: Request to batch
        """
        async with self._lock:
            if batch_key not in self._pending:
                self._pending[batch_key] = []

            self._pending[batch_key].append(request)

    async def get_batch(self, batch_key: str) -> Optional[list]:
        """
        Get batched requests if ready.

        Args:
            batch_key: Batch key

        Returns:
            Batched requests or None if not ready
        """
        async with self._lock:
            if batch_key in self._pending:
                batch = self._pending[batch_key]

                if len(batch) >= self.max_batch_size:
                    del self._pending[batch_key]
                    return batch

            return None

    async def flush_batch(self, batch_key: str) -> Optional[list]:
        """
        Flush batch regardless of size.

        Args:
            batch_key: Batch key

        Returns:
            Batched requests or None
        """
        async with self._lock:
            if batch_key in self._pending:
                batch = self._pending[batch_key]
                del self._pending[batch_key]
                return batch

            return None
