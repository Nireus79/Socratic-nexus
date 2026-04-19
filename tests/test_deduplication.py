"""Tests for request deduplication module."""

import asyncio
import pytest
from datetime import datetime, timedelta
from socratic_nexus.deduplication import RequestDeduplicator, CachedResponse, RequestBatcher


class TestCachedResponse:
    """Test CachedResponse dataclass."""

    def test_cached_response_creation(self):
        """Test creating a cached response."""
        response = CachedResponse(
            request_hash="test_hash",
            response={"result": "test"},
            ttl_seconds=3600,
        )
        assert response.request_hash == "test_hash"
        assert response.response == {"result": "test"}
        assert response.ttl_seconds == 3600
        assert response.hit_count == 0
        assert isinstance(response.created_at, datetime)

    def test_is_expired_false(self):
        """Test that fresh cache entry is not expired."""
        response = CachedResponse(
            request_hash="test_hash",
            response={"result": "test"},
            ttl_seconds=3600,
        )
        assert not response.is_expired()

    def test_is_expired_true(self):
        """Test that old cache entry is expired."""
        response = CachedResponse(
            request_hash="test_hash",
            response={"result": "test"},
            ttl_seconds=1,
            created_at=datetime.utcnow() - timedelta(seconds=2),
        )
        assert response.is_expired()


@pytest.mark.asyncio
class TestRequestDeduplicator:
    """Test RequestDeduplicator class."""

    async def test_initialization(self):
        """Test deduplicator initialization."""
        dedup = RequestDeduplicator(max_cache_size=500, default_ttl=7200)
        assert dedup.max_cache_size == 500
        assert dedup.default_ttl == 7200
        assert len(dedup._cache) == 0
        assert dedup._stats["total_requests"] == 0

    async def test_hash_request(self):
        """Test request hash computation."""
        dedup = RequestDeduplicator()

        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        temperature = 0.7

        hash1 = dedup._hash_request(model, messages, temperature=temperature)
        hash2 = dedup._hash_request(model, messages, temperature=temperature)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest length

    async def test_hash_request_different(self):
        """Test that different requests produce different hashes."""
        dedup = RequestDeduplicator()

        hash1 = dedup._hash_request(
            "gpt-4", [{"role": "user", "content": "Hello"}], temperature=0.7
        )
        hash2 = dedup._hash_request(
            "gpt-4", [{"role": "user", "content": "World"}], temperature=0.7
        )

        assert hash1 != hash2

    async def test_get_cached_response_miss(self):
        """Test getting non-existent cache entry."""
        dedup = RequestDeduplicator()
        result = await dedup.get_cached_response("gpt-4", [{"role": "user", "content": "Hello"}])
        assert result is None
        assert dedup._stats["cache_misses"] >= 1

    async def test_cache_and_retrieve(self):
        """Test caching and retrieving response."""
        dedup = RequestDeduplicator()

        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"result": "Hi there!"}

        await dedup.cache_response(model, messages, response)

        result = await dedup.get_cached_response(model, messages)
        assert result == response
        assert dedup._stats["cache_hits"] >= 1

    async def test_cache_hit_increments_count(self):
        """Test that cache hits increment hit count."""
        dedup = RequestDeduplicator()

        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"result": "Hi"}

        await dedup.cache_response(model, messages, response)

        # First get - hit
        await dedup.get_cached_response(model, messages)
        first_hits = dedup._stats["cache_hits"]

        # Second get - hit
        await dedup.get_cached_response(model, messages)
        second_hits = dedup._stats["cache_hits"]

        assert second_hits > first_hits

    async def test_expired_entry_not_returned(self):
        """Test that expired entries are not returned."""
        dedup = RequestDeduplicator()

        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"result": "Hi"}

        # Set with short TTL
        await dedup.cache_response(model, messages, response, ttl_seconds=1)

        # Simulate expiration
        request_hash = dedup._hash_request(model, messages)
        if request_hash in dedup._cache:
            dedup._cache[request_hash].created_at = datetime.utcnow() - timedelta(seconds=2)

        # Should not return expired entry
        result = await dedup.get_cached_response(model, messages)
        assert result is None

    async def test_clear_cache(self):
        """Test clearing the cache."""
        dedup = RequestDeduplicator()

        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"result": "Hi"}

        await dedup.cache_response(model, messages, response)
        assert len(dedup._cache) > 0

        await dedup.clear_cache()
        assert len(dedup._cache) == 0

    async def test_clear_old_entries(self):
        """Test clearing entries older than specified age."""
        dedup = RequestDeduplicator()

        model = "gpt-4"

        # Add new entry
        await dedup.cache_response(model, [{"role": "user", "content": "New"}], {"result": "new"})

        # Add old entry
        request_hash = dedup._hash_request(model, [{"role": "user", "content": "Old"}])
        old_response = CachedResponse(
            request_hash=request_hash,
            response={"result": "old"},
            created_at=datetime.utcnow() - timedelta(seconds=100),
        )
        dedup._cache[request_hash] = old_response

        # Clear entries older than 50 seconds
        removed = await dedup.clear_old_entries(max_age_seconds=50)

        assert removed >= 1

    async def test_cache_size_limit(self):
        """Test that cache respects size limit."""
        dedup = RequestDeduplicator(max_cache_size=2)

        # Add 3 items when max is 2
        await dedup.cache_response("gpt-4", [{"role": "user", "content": "1"}], {"result": "1"})
        await dedup.cache_response("gpt-4", [{"role": "user", "content": "2"}], {"result": "2"})
        await dedup.cache_response("gpt-4", [{"role": "user", "content": "3"}], {"result": "3"})

        # Should not exceed max size
        assert len(dedup._cache) <= dedup.max_cache_size

    async def test_get_stats(self):
        """Test statistics retrieval."""
        dedup = RequestDeduplicator()

        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"result": "Hi"}

        await dedup.cache_response(model, messages, response)
        await dedup.get_cached_response(model, messages)
        await dedup.get_cached_response(model, messages)
        await dedup.get_cached_response("gpt-4", [{"role": "user", "content": "World"}])

        stats = dedup.get_stats()

        assert stats["cache_hits"] >= 2
        assert stats["total_requests"] > 0
        assert "cache_misses" in stats

    async def test_get_cache_info(self):
        """Test cache info retrieval."""
        dedup = RequestDeduplicator()

        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"result": "Hi"}

        await dedup.cache_response(model, messages, response)

        info = dedup.get_cache_info()

        assert "total_entries" in info or "cache_size" in info
        assert "top_cached" in info or info.get("total_entries", 0) >= 0

    async def test_cache_with_cost(self):
        """Test caching response with cost tracking."""
        dedup = RequestDeduplicator()

        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        response = {"result": "Hi"}
        cost = 0.05

        await dedup.cache_response(model, messages, response, cost_usd=cost)

        stats = dedup.get_stats()
        assert "cost_saved_usd" in stats

    async def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        dedup = RequestDeduplicator()

        # Add entry with short TTL
        model = "gpt-4"
        messages = [{"role": "user", "content": "Test"}]
        await dedup.cache_response(model, messages, {"result": "test"}, ttl_seconds=1)

        request_hash = dedup._hash_request(model, messages)
        assert request_hash in dedup._cache

        # Simulate expiration
        dedup._cache[request_hash].created_at = datetime.utcnow() - timedelta(seconds=2)

        # Trigger cleanup
        await dedup._cleanup_expired()

        assert request_hash not in dedup._cache


@pytest.mark.asyncio
class TestRequestBatcher:
    """Test RequestBatcher class."""

    async def test_initialization(self):
        """Test batcher initialization."""
        batcher = RequestBatcher(batch_window_ms=100, max_batch_size=10)
        assert batcher.batch_window_ms == 100
        assert batcher.max_batch_size == 10

    async def test_add_request(self):
        """Test adding request to batch."""
        batcher = RequestBatcher()

        request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
        await batcher.add_request("batch_1", request)

        # Batch should be created
        assert "batch_1" in batcher._pending

    async def test_get_batch_not_ready(self):
        """Test getting batch before it's ready."""
        batcher = RequestBatcher(batch_window_ms=1000)

        request = {"model": "gpt-4", "messages": []}
        await batcher.add_request("batch_1", request)

        # Should not return before timeout
        batch = await asyncio.wait_for(batcher.get_batch("batch_1"), timeout=0.1)
        assert batch is None

    async def test_flush_batch(self):
        """Test flushing batch."""
        batcher = RequestBatcher()

        request1 = {"model": "gpt-4", "messages": []}
        request2 = {"model": "gpt-4", "messages": []}

        await batcher.add_request("batch_1", request1)
        await batcher.add_request("batch_1", request2)

        batch = await batcher.flush_batch("batch_1")

        assert batch is not None
        assert len(batch) == 2

    async def test_max_batch_size(self):
        """Test that batch is returned when max size is reached."""
        batcher = RequestBatcher(batch_window_ms=5000, max_batch_size=2)

        request1 = {"model": "gpt-4", "messages": []}
        request2 = {"model": "gpt-4", "messages": []}

        await batcher.add_request("batch_1", request1)
        await batcher.add_request("batch_1", request2)

        # Should return immediately when size reached
        batch = await asyncio.wait_for(batcher.get_batch("batch_1"), timeout=1.0)
        assert batch is not None
        assert len(batch) == 2
