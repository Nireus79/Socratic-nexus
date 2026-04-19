"""Tests for BaseProvider abstract base class.

Tests for provider functionality shared across all implementations:
- Cost calculation
- Usage tracking and callbacks
- Cache key generation
- Error handling
"""

import pytest
from unittest.mock import Mock
import hashlib

from socratic_nexus.models import LLMConfig, TokenUsage, PROVIDER_PRICING
from socratic_nexus.providers.base import BaseProvider
from socratic_nexus.exceptions import ProviderError


class ConcreteProvider(BaseProvider):
    """Concrete implementation for testing abstract base class."""

    def chat(self, message: str, **kwargs):
        """Minimal implementation for testing."""
        return Mock()

    async def achat(self, message: str, **kwargs):
        """Async version for testing."""
        return Mock()

    def stream(self, message: str, on_chunk, **kwargs):
        """Stream version for testing."""
        return Mock()

    async def astream(self, message: str, on_chunk, **kwargs):
        """Async stream version for testing."""
        return Mock()


class TestBaseProviderInitialization:
    """Test BaseProvider initialization."""

    def test_initialization_with_anthropic_config(self):
        """Test provider initializes with Anthropic config."""
        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="test-key"
        )
        provider = ConcreteProvider(config)

        assert provider.config == config
        assert provider.name == "ConcreteProvider"
        assert provider.pricing == PROVIDER_PRICING["anthropic"]
        assert provider._usage_callbacks == []

    def test_initialization_with_openai_config(self):
        """Test provider initializes with OpenAI config."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test-key")
        provider = ConcreteProvider(config)

        assert provider.config.provider == "openai"
        assert provider.pricing == PROVIDER_PRICING["openai"]

    def test_initialization_with_google_config(self):
        """Test provider initializes with Google config."""
        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="test-key")
        provider = ConcreteProvider(config)

        assert provider.config.provider == "google"
        assert provider.pricing == PROVIDER_PRICING["google"]

    def test_initialization_with_ollama_config(self):
        """Test provider initializes with Ollama config (local)."""
        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = ConcreteProvider(config)

        assert provider.config.provider == "ollama"
        assert provider.pricing == PROVIDER_PRICING["ollama"]

    def test_initialization_with_unknown_provider(self):
        """Test provider initializes with unknown provider (empty pricing)."""
        config = LLMConfig(provider="unknown", model="some-model")
        provider = ConcreteProvider(config)

        assert provider.pricing == {}


class TestCostCalculation:
    """Test cost calculation for token usage."""

    def test_calculate_cost_anthropic_haiku(self):
        """Test cost calculation for Anthropic Claude Haiku."""
        config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        provider = ConcreteProvider(config)

        # Haiku: $0.80 / 1M input, $4.00 / 1M output
        cost = provider.calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 4.80  # $0.80 + $4.00

    def test_calculate_cost_anthropic_sonnet(self):
        """Test cost calculation for Anthropic Claude Sonnet."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        # Sonnet: $3.00 / 1M input, $15.00 / 1M output
        cost = provider.calculate_cost(input_tokens=100_000, output_tokens=50_000)
        assert cost == 1.05  # (0.1 * 3.00) + (0.05 * 15.00) = 0.3 + 0.75

    def test_calculate_cost_anthropic_opus(self):
        """Test cost calculation for Anthropic Claude Opus."""
        config = LLMConfig(provider="anthropic", model="claude-opus-4-20250514")
        provider = ConcreteProvider(config)

        # Opus: $15.00 / 1M input, $75.00 / 1M output
        cost = provider.calculate_cost(input_tokens=500_000, output_tokens=500_000)
        assert cost == 45.00  # (0.5 * 15.00) + (0.5 * 75.00)

    def test_calculate_cost_openai_gpt4(self):
        """Test cost calculation for OpenAI GPT-4."""
        config = LLMConfig(provider="openai", model="gpt-4")
        provider = ConcreteProvider(config)

        # GPT-4: $30.00 / 1M input, $60.00 / 1M output
        cost = provider.calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 90.00  # $30.00 + $60.00

    def test_calculate_cost_google_gemini_pro(self):
        """Test cost calculation for Google Gemini Pro."""
        config = LLMConfig(provider="google", model="gemini-1.5-pro")
        provider = ConcreteProvider(config)

        # Gemini Pro: $1.25 / 1M input, $5.00 / 1M output
        cost = provider.calculate_cost(input_tokens=100_000, output_tokens=100_000)
        assert cost == 0.625  # (0.1 * 1.25) + (0.1 * 5.00)

    def test_calculate_cost_ollama_free(self):
        """Test cost calculation for Ollama (local, free)."""
        config = LLMConfig(provider="ollama", model="llama2")
        provider = ConcreteProvider(config)

        # Ollama is free
        cost = provider.calculate_cost(input_tokens=10_000_000, output_tokens=10_000_000)
        assert cost == 0.0

    def test_calculate_cost_with_custom_model(self):
        """Test cost calculation with explicit model parameter."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        # Calculate cost for different model than config
        cost = provider.calculate_cost(
            input_tokens=1_000_000, output_tokens=1_000_000, model="claude-haiku-4-5-20251001"
        )
        assert cost == 4.80  # Haiku pricing, not Sonnet

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        cost = provider.calculate_cost(input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model (should return 0)."""
        config = LLMConfig(provider="anthropic", model="unknown-model")
        provider = ConcreteProvider(config)

        cost = provider.calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 0.0


class TestUsageCallbacks:
    """Test usage callback registration and invocation."""

    def test_add_single_callback(self):
        """Test adding a single callback."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        callback = Mock()
        provider.add_usage_callback(callback)

        assert len(provider._usage_callbacks) == 1
        assert provider._usage_callbacks[0] == callback

    def test_add_multiple_callbacks(self):
        """Test adding multiple callbacks."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()

        provider.add_usage_callback(callback1)
        provider.add_usage_callback(callback2)
        provider.add_usage_callback(callback3)

        assert len(provider._usage_callbacks) == 3

    def test_notify_usage_single_callback(self):
        """Test invoking a single callback with usage."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        callback = Mock()
        provider.add_usage_callback(callback)

        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        provider._notify_usage(usage)

        callback.assert_called_once_with(usage)

    def test_notify_usage_multiple_callbacks(self):
        """Test invoking multiple callbacks with usage."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()

        provider.add_usage_callback(callback1)
        provider.add_usage_callback(callback2)
        provider.add_usage_callback(callback3)

        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        provider._notify_usage(usage)

        callback1.assert_called_once_with(usage)
        callback2.assert_called_once_with(usage)
        callback3.assert_called_once_with(usage)

    def test_notify_usage_callback_exception_ignored(self):
        """Test that exceptions in callbacks are ignored."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        callback_good = Mock()
        callback_bad = Mock(side_effect=ValueError("Callback error"))
        callback_after_bad = Mock()

        provider.add_usage_callback(callback_good)
        provider.add_usage_callback(callback_bad)
        provider.add_usage_callback(callback_after_bad)

        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        # Should not raise exception
        provider._notify_usage(usage)

        # Good callbacks should still be called
        callback_good.assert_called_once()
        callback_bad.assert_called_once()
        callback_after_bad.assert_called_once()

    def test_notify_usage_with_no_callbacks(self):
        """Test notifying with no callbacks registered."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        # Should not raise exception
        provider._notify_usage(usage)


class TestCacheKeyGeneration:
    """Test cache key generation for messages."""

    def test_generate_cache_key_simple_message(self):
        """Test cache key generation for simple message."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        message = "Hello, world!"
        cache_key = provider._generate_cache_key(message)

        # Should be SHA256 hex digest
        expected = hashlib.sha256(message.encode()).hexdigest()
        assert cache_key == expected
        assert len(cache_key) == 64  # SHA256 hex is 64 characters

    def test_generate_cache_key_identical_messages_identical_keys(self):
        """Test that identical messages produce identical cache keys."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        message = "This is a test message"
        key1 = provider._generate_cache_key(message)
        key2 = provider._generate_cache_key(message)

        assert key1 == key2

    def test_generate_cache_key_different_messages_different_keys(self):
        """Test that different messages produce different cache keys."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        key1 = provider._generate_cache_key("Message 1")
        key2 = provider._generate_cache_key("Message 2")

        assert key1 != key2

    def test_generate_cache_key_empty_message(self):
        """Test cache key generation for empty message."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        cache_key = provider._generate_cache_key("")
        expected = hashlib.sha256("".encode()).hexdigest()

        assert cache_key == expected

    def test_generate_cache_key_unicode_message(self):
        """Test cache key generation for Unicode message."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        message = "Hello, 世界! 🌍"
        cache_key = provider._generate_cache_key(message)
        expected = hashlib.sha256(message.encode()).hexdigest()

        assert cache_key == expected


class TestCreateTokenUsage:
    """Test TokenUsage object creation with cost calculation."""

    def test_create_token_usage_basic(self):
        """Test creating TokenUsage with basic values."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        usage = provider._create_token_usage(input_tokens=100, output_tokens=50)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.provider == "anthropic"
        assert usage.model == "claude-3-5-sonnet-20241022"
        assert usage.latency_ms == 0.0
        assert usage.cost_usd > 0.0  # Should calculate cost

    def test_create_token_usage_with_latency(self):
        """Test creating TokenUsage with latency measurement."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        usage = provider._create_token_usage(input_tokens=100, output_tokens=50, latency_ms=1234.5)

        assert usage.latency_ms == 1234.5

    def test_create_token_usage_cost_calculation(self):
        """Test that TokenUsage correctly calculates cost."""
        config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")  # $0.80 / $4.00
        provider = ConcreteProvider(config)

        usage = provider._create_token_usage(input_tokens=1_000_000, output_tokens=1_000_000)

        # Cost should be $0.80 + $4.00 = $4.80
        assert usage.cost_usd == 4.80

    def test_create_token_usage_ollama_free(self):
        """Test that Ollama TokenUsage has zero cost."""
        config = LLMConfig(provider="ollama", model="llama2")
        provider = ConcreteProvider(config)

        usage = provider._create_token_usage(input_tokens=100_000, output_tokens=100_000)

        assert usage.cost_usd == 0.0

    def test_create_token_usage_zero_tokens(self):
        """Test creating TokenUsage with zero tokens."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        usage = provider._create_token_usage(input_tokens=0, output_tokens=0)

        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.0

    def test_create_token_usage_has_timestamp(self):
        """Test that TokenUsage includes timestamp."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        usage = provider._create_token_usage(input_tokens=100, output_tokens=50)

        assert usage.timestamp is not None


class TestErrorHandling:
    """Test error handling in base provider."""

    def test_handle_api_error_generic(self):
        """Test generic error handling."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        original_error = ValueError("Something went wrong")

        with pytest.raises(ProviderError) as exc_info:
            provider._handle_api_error(original_error, "chat")

        assert "chat" in str(exc_info.value)
        assert "anthropic" in str(exc_info.value).lower()

    def test_handle_api_error_sets_error_code(self):
        """Test that error code is set correctly."""
        config = LLMConfig(provider="openai", model="gpt-4")
        provider = ConcreteProvider(config)

        original_error = RuntimeError("API Error")

        with pytest.raises(ProviderError) as exc_info:
            provider._handle_api_error(original_error, "stream")

        assert "OPENAI_ERROR" in str(exc_info.value.error_code)

    def test_handle_api_error_includes_operation(self):
        """Test that operation name is included in error."""
        config = LLMConfig(provider="google", model="gemini-1.5-pro")
        provider = ConcreteProvider(config)

        original_error = Exception("API call failed")

        with pytest.raises(ProviderError) as exc_info:
            provider._handle_api_error(original_error, "achat")

        assert "achat" in str(exc_info.value)


class TestProviderRepresentation:
    """Test provider string representation."""

    def test_repr_anthropic(self):
        """Test repr for Anthropic provider."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        repr_str = repr(provider)

        assert "ConcreteProvider" in repr_str
        assert "claude-3-5-sonnet-20241022" in repr_str

    def test_repr_openai(self):
        """Test repr for OpenAI provider."""
        config = LLMConfig(provider="openai", model="gpt-4")
        provider = ConcreteProvider(config)

        repr_str = repr(provider)

        assert "ConcreteProvider" in repr_str
        assert "gpt-4" in repr_str

    def test_repr_google(self):
        """Test repr for Google provider."""
        config = LLMConfig(provider="google", model="gemini-1.5-pro")
        provider = ConcreteProvider(config)

        repr_str = repr(provider)

        assert "ConcreteProvider" in repr_str
        assert "gemini-1.5-pro" in repr_str

    def test_repr_ollama(self):
        """Test repr for Ollama provider."""
        config = LLMConfig(provider="ollama", model="llama2")
        provider = ConcreteProvider(config)

        repr_str = repr(provider)

        assert "ConcreteProvider" in repr_str
        assert "llama2" in repr_str


class TestIntegrationCostWithCallbacks:
    """Integration tests combining cost calculation and callbacks."""

    def test_create_usage_and_notify_callbacks(self):
        """Test creating TokenUsage and notifying callbacks."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
        provider = ConcreteProvider(config)

        # Add callback to track usage
        received_usages = []

        def track_usage(usage):
            received_usages.append(usage)

        provider.add_usage_callback(track_usage)

        # Create usage and notify
        usage = provider._create_token_usage(100, 50)
        provider._notify_usage(usage)

        # Verify callback received correct usage
        assert len(received_usages) == 1
        assert received_usages[0].cost_usd > 0.0
        assert received_usages[0].total_tokens == 150

    def test_multiple_usages_accumulate(self):
        """Test that multiple usages accumulate correctly."""
        config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
        provider = ConcreteProvider(config)

        total_cost = 0.0
        total_input = 0
        total_output = 0

        def track_totals(usage):
            nonlocal total_cost, total_input, total_output
            total_cost += usage.cost_usd
            total_input += usage.input_tokens
            total_output += usage.output_tokens

        provider.add_usage_callback(track_totals)

        # Multiple requests (100 input, 50 output tokens each)
        # Haiku: $0.80/1M input, $4.00/1M output
        # Per request: (100/1M)*0.80 + (50/1M)*4.00 = 0.00008 + 0.0002 = 0.00028
        for i in range(3):
            usage = provider._create_token_usage(100, 50)
            provider._notify_usage(usage)

        # Verify accumulation
        assert total_input == 300
        assert total_output == 150
        assert total_cost == pytest.approx(3 * 0.00028, rel=0.01)  # 3 requests at 0.00028 each
