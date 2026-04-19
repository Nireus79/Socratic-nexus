"""Tests for LLMClient."""

import pytest
from socratic_nexus import LLMClient, LLMConfig
from socratic_nexus.exceptions import ConfigurationError


def test_client_initialization():
    """Test LLMClient initialization."""
    config = LLMConfig(provider="anthropic", model="claude-opus", api_key="test-key")
    client = LLMClient(config=config)
    assert client.config.provider == "anthropic"
    assert client.config.model == "claude-opus"


def test_client_initialization_from_kwargs():
    """Test LLMClient initialization from kwargs."""
    client = LLMClient(provider="openai", model="gpt-4", api_key="test-key")
    assert client.config.provider == "openai"
    assert client.config.model == "gpt-4"


def test_client_initialization_no_provider():
    """Test LLMClient initialization without provider raises error."""
    with pytest.raises(ConfigurationError):
        LLMClient(model="gpt-4")


def test_usage_stats():
    """Test usage stats tracking."""
    config = LLMConfig(provider="anthropic", model="claude-opus", api_key="test-key")
    client = LLMClient(config=config)
    stats = client.get_usage_stats()
    assert stats.total_requests == 0
    assert stats.total_cost_usd == 0.0


def test_client_provider_aliases():
    """Test provider aliases work correctly."""
    # These should work without error - aliases may be resolved at instantiation time
    client1 = LLMClient(provider="claude", model="claude-opus", api_key="test")
    assert client1.config.provider in ["claude", "anthropic"]

    # OpenAI aliases
    client2 = LLMClient(provider="gpt", model="gpt-4", api_key="test")
    assert client2.config.provider in ["gpt", "openai"]

    # Google aliases
    client3 = LLMClient(provider="gemini", model="gemini-pro", api_key="test")
    assert client3.config.provider in ["gemini", "google"]

    # Ollama aliases
    client4 = LLMClient(provider="local", model="llama2")
    assert client4.config.provider in ["local", "ollama"]


def test_client_config_temperature():
    """Test temperature configuration."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        temperature=0.5,
    )
    assert config.temperature == 0.5


def test_client_config_max_tokens():
    """Test max_tokens configuration."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        max_tokens=512,
    )
    assert config.max_tokens == 512


def test_client_retry_config():
    """Test retry configuration."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        retry_attempts=5,
        retry_backoff_factor=3.0,
    )
    assert config.retry_attempts == 5
    assert config.retry_backoff_factor == 3.0


def test_client_cache_config():
    """Test cache configuration."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        cache_responses=True,
        cache_ttl=600,
    )
    assert config.cache_responses is True
    assert config.cache_ttl == 600


def test_client_add_usage_callback():
    """Test adding usage callbacks."""
    config = LLMConfig(provider="anthropic", model="claude-opus", api_key="test-key")
    client = LLMClient(config=config)

    def test_callback(usage):
        pass

    # Should not raise an error
    client.add_usage_callback(test_callback)
    # Callback was registered (actual invocation requires real API call)


def test_client_cache_disabled():
    """Test client with response caching disabled."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        cache_responses=False,
    )
    client = LLMClient(config=config)
    assert client._cache is None


def test_client_cache_enabled():
    """Test client with response caching enabled."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        cache_responses=True,
    )
    client = LLMClient(config=config)
    assert client._cache is not None
