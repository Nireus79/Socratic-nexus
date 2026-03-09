"""Tests for AsyncLLMClient."""

import pytest
import asyncio
from socrates_nexus import AsyncLLMClient, LLMConfig
from socrates_nexus.exceptions import ConfigurationError


@pytest.mark.asyncio
async def test_async_client_initialization():
    """Test AsyncLLMClient initialization."""
    config = LLMConfig(provider="anthropic", model="claude-opus", api_key="test-key")
    client = AsyncLLMClient(config=config)
    assert client.config.provider == "anthropic"
    assert client.config.model == "claude-opus"


@pytest.mark.asyncio
async def test_async_client_initialization_from_kwargs():
    """Test AsyncLLMClient initialization from kwargs."""
    client = AsyncLLMClient(provider="openai", model="gpt-4", api_key="test-key")
    assert client.config.provider == "openai"
    assert client.config.model == "gpt-4"


@pytest.mark.asyncio
async def test_async_client_initialization_no_provider():
    """Test AsyncLLMClient initialization without provider raises error."""
    with pytest.raises(ConfigurationError):
        AsyncLLMClient(model="gpt-4")


@pytest.mark.asyncio
async def test_async_client_get_usage_stats():
    """Test AsyncLLMClient usage stats."""
    config = LLMConfig(provider="anthropic", model="claude-opus", api_key="test-key")
    client = AsyncLLMClient(config=config)
    stats = client.get_usage_stats()
    assert stats.total_requests == 0
    assert stats.total_cost_usd == 0.0


@pytest.mark.asyncio
async def test_async_client_add_callback():
    """Test adding callbacks to AsyncLLMClient."""
    config = LLMConfig(provider="anthropic", model="claude-opus", api_key="test-key")
    client = AsyncLLMClient(config=config)

    callback_called = []

    def test_callback(usage):
        callback_called.append(usage)

    client.add_usage_callback(test_callback)
    # Verify callback was registered
    assert len(client._usage_callbacks) > 0


@pytest.mark.asyncio
async def test_async_client_provider_aliases():
    """Test provider aliases work with AsyncLLMClient."""
    # Claude aliases
    client1 = AsyncLLMClient(provider="claude", model="claude-opus", api_key="test")
    assert client1.config.provider == "anthropic"

    # OpenAI aliases
    client2 = AsyncLLMClient(provider="gpt", model="gpt-4", api_key="test")
    assert client2.config.provider == "openai"


@pytest.mark.asyncio
async def test_async_client_config_temperature():
    """Test AsyncLLMClient temperature configuration."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        temperature=0.5,
    )
    client = AsyncLLMClient(config=config)
    assert client.config.temperature == 0.5


@pytest.mark.asyncio
async def test_async_client_config_max_tokens():
    """Test AsyncLLMClient max_tokens configuration."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        max_tokens=512,
    )
    client = AsyncLLMClient(config=config)
    assert client.config.max_tokens == 512


@pytest.mark.asyncio
async def test_async_client_retry_config():
    """Test AsyncLLMClient retry configuration."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        retry_attempts=5,
        retry_backoff_factor=3.0,
    )
    client = AsyncLLMClient(config=config)
    assert client.config.retry_attempts == 5
    assert client.config.retry_backoff_factor == 3.0


@pytest.mark.asyncio
async def test_async_client_cache_config():
    """Test AsyncLLMClient cache configuration."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        cache_responses=True,
        cache_ttl=600,
    )
    client = AsyncLLMClient(config=config)
    assert client.config.cache_responses is True
    assert client.config.cache_ttl == 600


def test_async_client_concurrent_usage():
    """Test that multiple AsyncLLMClients can be used concurrently."""

    async def run_test():
        client1 = AsyncLLMClient(provider="anthropic", model="claude-opus", api_key="test")
        client2 = AsyncLLMClient(provider="openai", model="gpt-4", api_key="test")
        client3 = AsyncLLMClient(provider="google", model="gemini-pro", api_key="test")

        # All should initialize without conflict
        assert client1.config.provider == "anthropic"
        assert client2.config.provider == "openai"
        assert client3.config.provider == "google"

    asyncio.run(run_test())


@pytest.mark.asyncio
async def test_async_client_config_defaults():
    """Test AsyncLLMClient default configuration values."""
    client = AsyncLLMClient(provider="anthropic", model="claude-opus", api_key="test")

    assert client.config.temperature == 0.7
    assert client.config.max_tokens == 1024
    assert client.config.retry_attempts == 3
    assert client.config.retry_backoff_factor == 2.0
    assert client.config.request_timeout == 60
    assert client.config.cache_responses is False
