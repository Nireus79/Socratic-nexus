"""Tests for response caching functionality."""

import pytest
import time
from socrates_nexus.utils.cache import TTLCache


def test_ttl_cache_initialization():
    """Test TTLCache initialization."""
    cache = TTLCache(ttl_seconds=300)

    assert cache.ttl_seconds == 300
    assert len(cache) == 0


def test_ttl_cache_set_and_get():
    """Test setting and getting values from cache."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_ttl_cache_multiple_items():
    """Test storing multiple items in cache."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_ttl_cache_get_nonexistent():
    """Test getting nonexistent key returns None."""
    cache = TTLCache(ttl_seconds=300)

    assert cache.get("nonexistent") is None


def test_ttl_cache_overwrite():
    """Test overwriting existing key."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("key", "value1")
    assert cache.get("key") == "value1"

    cache.set("key", "value2")
    assert cache.get("key") == "value2"


def test_ttl_cache_expiry():
    """Test that items expire after TTL."""
    cache = TTLCache(ttl_seconds=1)

    cache.set("key", "value")
    assert cache.get("key") == "value"

    # Wait for expiry
    time.sleep(1.1)

    assert cache.get("key") is None


def test_ttl_cache_length():
    """Test cache length tracking."""
    cache = TTLCache(ttl_seconds=300)

    assert len(cache) == 0

    cache.set("key1", "value1")
    assert len(cache) == 1

    cache.set("key2", "value2")
    assert len(cache) == 2


def test_ttl_cache_clear():
    """Test clearing the cache."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    assert len(cache) == 2

    cache.clear()
    assert len(cache) == 0
    assert cache.get("key1") is None


def test_ttl_cache_contains():
    """Test checking if key exists in cache."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("key", "value")
    assert "key" in cache
    assert "nonexistent" not in cache


def test_ttl_cache_delete():
    """Test deleting a key from cache."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("key", "value")
    assert cache.get("key") == "value"

    del cache["key"]
    assert cache.get("key") is None


def test_ttl_cache_values():
    """Test getting all values from cache."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("key1", "value1")
    cache.set("key2", "value2")

    values = list(cache.values())
    assert len(values) == 2
    assert "value1" in values
    assert "value2" in values


def test_ttl_cache_keys():
    """Test getting all keys from cache."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("key1", "value1")
    cache.set("key2", "value2")

    keys = list(cache.keys())
    assert len(keys) == 2
    assert "key1" in keys
    assert "key2" in keys


def test_ttl_cache_items():
    """Test getting all items from cache."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("key1", "value1")
    cache.set("key2", "value2")

    items = list(cache.items())
    assert len(items) == 2
    assert ("key1", "value1") in items
    assert ("key2", "value2") in items


def test_ttl_cache_with_different_types():
    """Test cache with different value types."""
    cache = TTLCache(ttl_seconds=300)

    cache.set("str_key", "string value")
    cache.set("int_key", 42)
    cache.set("list_key", [1, 2, 3])
    cache.set("dict_key", {"nested": "dict"})

    assert cache.get("str_key") == "string value"
    assert cache.get("int_key") == 42
    assert cache.get("list_key") == [1, 2, 3]
    assert cache.get("dict_key") == {"nested": "dict"}


def test_ttl_cache_partial_expiry():
    """Test that only expired items are removed."""
    cache = TTLCache(ttl_seconds=1)

    cache.set("key1", "value1")
    time.sleep(0.5)

    cache.set("key2", "value2")
    time.sleep(0.6)

    # key1 should be expired, key2 should not
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"


def test_ttl_cache_refresh_on_access():
    """Test that accessing a value doesn't extend TTL."""
    cache = TTLCache(ttl_seconds=1)

    cache.set("key", "value")
    time.sleep(0.5)

    # Access the value
    assert cache.get("key") == "value"

    time.sleep(0.6)

    # Should still be expired
    assert cache.get("key") is None


def test_ttl_cache_thread_safety():
    """Test that cache is thread-safe (basic check)."""
    import threading

    cache = TTLCache(ttl_seconds=300)

    def write_thread():
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")

    def read_thread():
        for i in range(10):
            cache.get(f"key_{i}")

    # Run threads
    t1 = threading.Thread(target=write_thread)
    t2 = threading.Thread(target=read_thread)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Cache should have data
    assert len(cache) > 0


def test_ttl_cache_zero_ttl():
    """Test cache with very short TTL."""
    cache = TTLCache(ttl_seconds=0.01)

    cache.set("key", "value")
    assert cache.get("key") == "value"

    time.sleep(0.02)

    assert cache.get("key") is None
