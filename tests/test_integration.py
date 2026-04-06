"""
Integration tests for Socrates Nexus with real LLM APIs.

These tests require valid API keys and will make actual API calls.
Run with: pytest tests/test_integration.py -v --requires_api

Set environment variables:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- GOOGLE_API_KEY
"""

import os
import pytest
from socrates_nexus import LLMClient, AsyncLLMClient

pytestmark = pytest.mark.requires_api


@pytest.fixture
def anthropic_client():
    """Create Anthropic client if API key is available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return LLMClient(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
    )


@pytest.fixture
def openai_client():
    """Create OpenAI client if API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return LLMClient(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key=api_key,
    )


@pytest.fixture
def google_client():
    """Create Google client if API key is available."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    return LLMClient(
        provider="google",
        model="gemini-1.5-flash",
        api_key=api_key,
    )


def test_anthropic_chat(anthropic_client):
    """Test actual Claude API call."""
    response = anthropic_client.chat("What is 2+2?")

    assert response.content is not None
    assert len(response.content) > 0
    assert response.usage.total_tokens > 0
    assert response.usage.cost_usd >= 0


def test_anthropic_streaming(anthropic_client):
    """Test Claude streaming."""
    chunks = []

    def on_chunk(chunk):
        chunks.append(chunk)

    response = anthropic_client.stream(
        "Count to 3",
        on_chunk=on_chunk,
    )

    assert len(chunks) > 0
    assert "".join(chunks) in response.content or response.content in "".join(chunks)
    assert response.usage.output_tokens > 0


def test_openai_chat(openai_client):
    """Test actual OpenAI API call."""
    response = openai_client.chat("What is 2+2?")

    assert response.content is not None
    assert len(response.content) > 0
    assert response.usage.total_tokens > 0
    assert response.usage.cost_usd >= 0


def test_openai_streaming(openai_client):
    """Test OpenAI streaming."""
    chunks = []

    def on_chunk(chunk):
        chunks.append(chunk)

    response = openai_client.stream(
        "Count to 3",
        on_chunk=on_chunk,
    )

    assert len(chunks) > 0
    assert response.usage.output_tokens > 0


def test_google_chat(google_client):
    """Test actual Google Gemini API call."""
    response = google_client.chat("What is 2+2?")

    assert response.content is not None
    assert len(response.content) > 0
    assert response.usage.total_tokens > 0


def test_cost_calculation(anthropic_client):
    """Test that cost is calculated correctly."""
    response = anthropic_client.chat("Hello")

    # Cost should be calculated
    assert response.usage.cost_usd >= 0

    # Get stats
    stats = anthropic_client.get_usage_stats()
    assert stats.total_requests > 0
    assert stats.total_cost_usd >= 0


def test_error_handling_invalid_api_key():
    """Test error handling with invalid API key."""
    from socrates_nexus.exceptions import LLMError

    client = LLMClient(
        provider="anthropic",
        model="claude-opus",
        api_key="invalid-key-12345",
    )

    with pytest.raises(LLMError):
        client.chat("Hello")


@pytest.mark.asyncio
async def test_async_anthropic_chat():
    """Test async Claude API call."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    client = AsyncLLMClient(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
    )

    response = await client.chat("What is 2+2?")

    assert response.content is not None
    assert response.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_async_concurrent_requests():
    """Test concurrent async requests."""
    import asyncio

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    client = AsyncLLMClient(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        api_key=api_key,
    )

    responses = await asyncio.gather(
        client.chat("What is 2+2?"),
        client.chat("What is 3+3?"),
        client.chat("What is 4+4?"),
    )

    assert len(responses) == 3
    assert all(r.content for r in responses)
    assert sum(r.usage.total_tokens for r in responses) > 0


def test_multi_provider_fallback():
    """Test fallback between providers."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    providers = [
        {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "api_key": api_key,
        },
    ]

    for config in providers:
        try:
            client = LLMClient(**config)
            response = client.chat("Hello")
            assert response.content
            return
        except Exception:
            continue

    pytest.fail("All providers failed")


def test_response_caching(anthropic_client):
    """Test response caching functionality."""
    client = LLMClient(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        cache_responses=True,
        cache_ttl=300,
    )

    message = "What is the capital of France?"

    # First request - hits API
    response1 = client.chat(message)
    stats1 = client.get_usage_stats()
    cost1 = stats1.total_cost_usd

    # Second request - should hit cache
    response2 = client.chat(message)
    stats2 = client.get_usage_stats()
    cost2 = stats2.total_cost_usd

    assert response1.content == response2.content
    # Cost should not increase significantly (only one API call)
    assert cost2 <= cost1 * 1.1  # Allow 10% margin
