"""Pytest configuration for Socrates Nexus tests."""

import pytest
import os
from unittest.mock import MagicMock
from socrates_nexus.models import TokenUsage, ChatResponse


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    return os.getenv("ANTHROPIC_API_KEY")


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture
def google_api_key():
    """Get Google API key from environment."""
    return os.getenv("GOOGLE_API_KEY")


@pytest.fixture
def mock_usage():
    """Create a mock TokenUsage object."""
    return TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="anthropic",
        model="claude-opus",
        cost_usd=0.015,
        latency_ms=250.0,
    )


@pytest.fixture
def mock_response(mock_usage):
    """Create a mock ChatResponse object."""
    return ChatResponse(
        content="This is a mock response.",
        usage=mock_usage,
        model="claude-opus",
        provider="anthropic",
    )


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.chat.return_value = ChatResponse(
        content="Mock response",
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            provider="mock",
            model="mock-model",
            cost_usd=0.001,
        ),
        model="mock-model",
        provider="mock",
    )
    return provider
