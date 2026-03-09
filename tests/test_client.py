"""Tests for LLMClient."""

import pytest
from socrates_nexus import LLMClient, LLMConfig
from socrates_nexus.exceptions import ConfigurationError


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
