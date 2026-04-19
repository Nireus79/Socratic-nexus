"""Tests for data models and cost calculation."""

from datetime import datetime
from socratic_nexus.models import TokenUsage, ChatResponse, UsageStats, LLMConfig
from socratic_nexus.models import PROVIDER_PRICING


def test_token_usage_creation():
    """Test TokenUsage creation and attributes."""
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="anthropic",
        model="claude-opus",
        cost_usd=0.015,
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
    assert usage.provider == "anthropic"
    assert usage.model == "claude-opus"
    assert usage.cost_usd == 0.015


def test_token_usage_cost_calculation_anthropic_haiku():
    """Test cost calculation for Anthropic Haiku."""
    # Haiku: $0.80 per 1M input, $4.00 per 1M output
    # Expected: (1000/1000000 * 0.80) + (1000/1000000 * 4.00) = 0.0048
    pricing = PROVIDER_PRICING["anthropic"]["claude-haiku-4-5-20251001"]
    expected_cost = (1000 / 1000000 * pricing["input"]) + (1000 / 1000000 * pricing["output"])

    usage = TokenUsage(
        input_tokens=1000,  # 0.001M tokens
        output_tokens=1000,  # 0.001M tokens
        total_tokens=2000,
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        cost_usd=expected_cost,
    )

    assert usage.cost_usd > 0
    assert 0.004 < usage.cost_usd < 0.005


def test_token_usage_cost_calculation_anthropic_sonnet():
    """Test cost calculation for Anthropic Sonnet."""
    # Sonnet: $3.00 per 1M input, $15.00 per 1M output
    pricing = PROVIDER_PRICING["anthropic"]["claude-3-5-sonnet-20241022"]
    expected_cost = (1000 / 1000000 * pricing["input"]) + (1000 / 1000000 * pricing["output"])

    usage = TokenUsage(
        input_tokens=1000,
        output_tokens=1000,
        total_tokens=2000,
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        cost_usd=expected_cost,
    )

    # Expected: (1000/1000000 * 3.00) + (1000/1000000 * 15.00) = 0.003 + 0.015 = 0.018
    assert usage.cost_usd > 0
    assert 0.017 < usage.cost_usd < 0.019


def test_token_usage_zero_cost_ollama():
    """Test that Ollama (local) has zero cost."""
    usage = TokenUsage(
        input_tokens=1000,
        output_tokens=1000,
        total_tokens=2000,
        provider="ollama",
        model="llama2",
    )

    assert usage.cost_usd == 0.0


def test_chat_response_creation():
    """Test ChatResponse creation."""
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="anthropic",
        model="claude-opus",
        cost_usd=0.015,
    )

    response = ChatResponse(
        content="This is a response",
        usage=usage,
        model="claude-opus",
        provider="anthropic",
    )

    assert response.content == "This is a response"
    assert response.usage.total_tokens == 150
    assert response.model == "claude-opus"
    assert response.provider == "anthropic"


def test_chat_response_with_usage_timestamp():
    """Test ChatResponse usage includes timestamp."""
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="anthropic",
        model="claude-opus",
        cost_usd=0.015,
    )

    response = ChatResponse(
        content="Response",
        usage=usage,
        model="claude-opus",
        provider="anthropic",
    )

    assert response.usage.timestamp is not None
    assert isinstance(response.usage.timestamp, datetime)


def test_usage_stats_initialization():
    """Test UsageStats initialization."""
    stats = UsageStats()

    assert stats.total_requests == 0
    assert stats.total_input_tokens == 0
    assert stats.total_output_tokens == 0
    assert stats.total_cost_usd == 0.0
    assert len(stats.by_provider) == 0
    assert len(stats.by_model) == 0


def test_llm_config_validation():
    """Test LLMConfig validation."""
    # Valid config
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
    )
    assert config.provider == "anthropic"

    # Temperature bounds
    config_temp = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
        temperature=0.7,
    )
    assert 0.0 <= config_temp.temperature <= 2.0


def test_llm_config_defaults():
    """Test LLMConfig default values."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus",
        api_key="test-key",
    )

    assert config.temperature == 0.7
    assert config.max_tokens is None
    assert config.retry_attempts == 3
    assert config.retry_backoff_factor == 2.0
    assert config.timeout == 30
    assert config.cache_responses is True
    assert config.cache_ttl == 300


def test_provider_pricing_exists():
    """Test that all providers have pricing defined."""
    assert "anthropic" in PROVIDER_PRICING
    assert "openai" in PROVIDER_PRICING
    assert "google" in PROVIDER_PRICING
    assert "ollama" in PROVIDER_PRICING


def test_provider_pricing_anthropic():
    """Test Anthropic pricing data."""
    pricing = PROVIDER_PRICING["anthropic"]

    # Check that we have pricing for known models
    assert "claude-haiku-4-5-20251001" in pricing
    assert "claude-3-5-sonnet-20241022" in pricing
    assert len(pricing) > 0

    # Check structure
    haiku_pricing = pricing["claude-haiku-4-5-20251001"]
    assert "input" in haiku_pricing
    assert "output" in haiku_pricing
    assert haiku_pricing["input"] > 0
    assert haiku_pricing["output"] > 0


def test_provider_pricing_openai():
    """Test OpenAI pricing data."""
    pricing = PROVIDER_PRICING["openai"]

    # Check that we have pricing for known models
    assert "gpt-4" in pricing or "gpt-3.5-turbo" in pricing

    # Check structure if models exist
    if "gpt-3.5-turbo" in pricing:
        gpt_pricing = pricing["gpt-3.5-turbo"]
        assert "input" in gpt_pricing
        assert "output" in gpt_pricing


def test_provider_pricing_ollama_free():
    """Test that Ollama pricing is free."""
    pricing = PROVIDER_PRICING["ollama"]

    # All Ollama models should be free
    for model_pricing in pricing.values():
        assert model_pricing["input"] == 0
        assert model_pricing["output"] == 0


def test_token_usage_with_latency():
    """Test TokenUsage with latency."""
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="anthropic",
        model="claude-opus",
        cost_usd=0.015,
        latency_ms=250.5,
    )

    assert usage.latency_ms == 250.5


def test_token_usage_timestamp():
    """Test TokenUsage includes timestamp."""
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="anthropic",
        model="claude-opus",
        cost_usd=0.015,
    )

    assert usage.timestamp is not None
    assert isinstance(usage.timestamp, datetime)


def test_llm_config_base_url():
    """Test LLMConfig base_url for custom endpoints."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="test-key",
        base_url="http://localhost:8000",
    )

    assert config.base_url == "http://localhost:8000"
