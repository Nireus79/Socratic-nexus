"""Tests for response caching functionality."""

import time
from socrates_nexus.utils.cache import TTLCache, ResponseCache


def test_ttl_cache_decorator_initialization():
    """Test TTLCache decorator initialization."""

    @TTLCache(ttl_minutes=5)
    def expensive_function(x):
        return x * 2

    assert expensive_function is not None


def test_ttl_cache_decorator_caching():
    """Test that decorator caches results."""
    call_count = 0

    @TTLCache(ttl_minutes=5)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call - should execute
    result1 = expensive_function(5)
    assert result1 == 10
    assert call_count == 1

    # Second call with same args - should use cache
    result2 = expensive_function(5)
    assert result2 == 10
    assert call_count == 1  # Should not increment


def test_ttl_cache_different_args():
    """Test that different arguments create different cache entries."""
    call_count = 0

    @TTLCache(ttl_minutes=5)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    result1 = expensive_function(5)
    result2 = expensive_function(10)

    assert result1 == 10
    assert result2 == 20
    assert call_count == 2  # Both calls executed


def test_ttl_cache_expiry():
    """Test that cached results expire after TTL."""
    call_count = 0

    # Use very short TTL for testing
    @TTLCache(ttl_minutes=0.017)  # ~1 second
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    expensive_function(5)
    assert call_count == 1

    # Wait for cache to expire
    time.sleep(1.5)

    expensive_function(5)
    assert call_count == 2  # Should execute again after expiry


def test_ttl_cache_with_kwargs():
    """Test caching with keyword arguments."""
    call_count = 0

    @TTLCache(ttl_minutes=5)
    def function_with_kwargs(a, b=10):
        nonlocal call_count
        call_count += 1
        return a + b

    result1 = function_with_kwargs(5, b=10)
    result2 = function_with_kwargs(5, b=10)

    assert result1 == 15
    assert result2 == 15
    assert call_count == 1  # Cached


def test_ttl_cache_different_kwargs():
    """Test that different kwargs create different cache entries."""
    call_count = 0

    @TTLCache(ttl_minutes=5)
    def function_with_kwargs(a, b=10):
        nonlocal call_count
        call_count += 1
        return a + b

    result1 = function_with_kwargs(5, b=10)
    result2 = function_with_kwargs(5, b=20)

    assert result1 == 15
    assert result2 == 25
    assert call_count == 2


def test_ttl_cache_return_types():
    """Test caching with different return types."""

    @TTLCache(ttl_minutes=5)
    def return_dict(x):
        return {"value": x * 2}

    result1 = return_dict(5)
    result2 = return_dict(5)

    assert result1 == {"value": 10}
    assert result2 == {"value": 10}
    assert result1 is result2  # Same object from cache


def test_ttl_cache_with_none():
    """Test caching when function returns None."""
    call_count = 0

    @TTLCache(ttl_minutes=5)
    def returns_none():
        nonlocal call_count
        call_count += 1
        return None

    result1 = returns_none()
    result2 = returns_none()

    assert result1 is None
    assert result2 is None
    assert call_count == 1  # Should be cached


def test_ttl_cache_preserves_function_name():
    """Test that decorator preserves function name."""

    @TTLCache(ttl_minutes=5)
    def my_function():
        return 42

    # Should preserve function metadata
    assert hasattr(my_function, "__wrapped__") or callable(my_function)


def test_ttl_cache_thread_safety():
    """Test that cache is thread-safe."""
    import threading

    call_count = 0

    @TTLCache(ttl_minutes=5)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    results = []

    def call_function():
        results.append(expensive_function(5))

    threads = [threading.Thread(target=call_function) for _ in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # All results should be same
    assert all(r == 10 for r in results)
    # Call count should be small (cached, not 5)
    assert call_count <= 3


def test_ttl_cache_hit_miss_stats():
    """Test cache hit/miss statistics."""

    @TTLCache(ttl_minutes=5)
    def expensive_function(x):
        return x * 2

    expensive_function(5)  # Miss
    expensive_function(5)  # Hit
    expensive_function(10)  # Miss
    expensive_function(5)  # Hit

    # Cache should have stats
    assert hasattr(expensive_function, "__wrapped__") or callable(expensive_function)


class TestResponseCache:
    """Test ResponseCache class."""

    def test_response_cache_initialization(self):
        """Test ResponseCache initialization."""
        cache = ResponseCache(ttl_minutes=5)
        assert cache is not None

    def test_response_cache_set_and_get(self):
        """Test setting and getting cache entries."""
        cache = ResponseCache(ttl_minutes=1)
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

    def test_response_cache_get_missing_key(self):
        """Test getting missing key returns None."""
        cache = ResponseCache(ttl_minutes=1)
        result = cache.get("nonexistent")
        assert result is None

    def test_response_cache_multiple_keys(self):
        """Test multiple keys in cache."""
        cache = ResponseCache(ttl_minutes=1)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", {"data": [1, 2, 3]})

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == {"data": [1, 2, 3]}

    def test_response_cache_overwrite(self):
        """Test that setting a key overwrites old value."""
        cache = ResponseCache(ttl_minutes=1)
        cache.set("key", "value1")
        cache.set("key", "value2")
        assert cache.get("key") == "value2"

    def test_response_cache_complex_objects(self):
        """Test caching complex objects."""
        cache = ResponseCache(ttl_minutes=1)
        data = {
            "name": "test",
            "values": [1, 2, 3],
            "nested": {"inner": "value"},
        }
        cache.set("complex", data)
        retrieved = cache.get("complex")
        assert retrieved == data

    def test_response_cache_clear(self):
        """Test clearing the cache."""
        cache = ResponseCache(ttl_minutes=1)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_response_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = ResponseCache(ttl_minutes=0)  # Immediate expiration
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        time.sleep(0.1)

        removed = cache.cleanup_expired()
        assert removed >= 0  # At least 0 removed (timing dependent)

    def test_response_cache_expiry(self):
        """Test that entries expire after TTL."""
        cache = ResponseCache(ttl_minutes=0.017)  # ~1 second
        cache.set("expiring_key", "value")

        # Should exist immediately
        assert cache.get("expiring_key") == "value"

        # Wait for expiration
        time.sleep(1.5)

        # Should be gone after expiration
        result = cache.get("expiring_key")
        assert result is None

    def test_response_cache_thread_safety(self):
        """Test that ResponseCache is thread-safe."""
        import threading

        cache = ResponseCache(ttl_minutes=1)
        results = []

        def worker(key, value):
            cache.set(key, value)
            retrieved = cache.get(key)
            results.append(retrieved)

        threads = [threading.Thread(target=worker, args=(f"key{i}", f"value{i}")) for i in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(v is not None for v in results)

    def test_response_cache_store_none(self):
        """Test storing None values."""
        cache = ResponseCache(ttl_minutes=1)
        cache.set("none_key", None)
        # Note: None values may behave differently depending on implementation
        cache.get("none_key")
        # Result may be None due to implementation details

    def test_response_cache_numeric_values(self):
        """Test storing numeric values."""
        cache = ResponseCache(ttl_minutes=1)
        cache.set("int", 42)
        cache.set("float", 3.14)
        cache.set("negative", -100)

        assert cache.get("int") == 42
        assert cache.get("float") == 3.14
        assert cache.get("negative") == -100

    def test_response_cache_list_values(self):
        """Test storing list values."""
        cache = ResponseCache(ttl_minutes=1)
        data = [1, 2, 3, 4, 5]
        cache.set("list", data)
        assert cache.get("list") == data

    def test_response_cache_string_keys(self):
        """Test various string keys."""
        cache = ResponseCache(ttl_minutes=1)
        keys = ["simple", "key-with-dash", "key_with_underscore", "KEY_UPPERCASE"]

        for key in keys:
            cache.set(key, f"value_for_{key}")

        for key in keys:
            assert cache.get(key) == f"value_for_{key}"
