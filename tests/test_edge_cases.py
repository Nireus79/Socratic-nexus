"""Tests for edge cases and configuration validation."""

import pytest
from unittest.mock import patch
from socratic_nexus.models import LLMConfig, ChatResponse, TokenUsage
from socratic_nexus.client import LLMClient
from socratic_nexus.exceptions import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
)


class TestConfigurationValidation:
    """Test configuration validation and bounds checking."""

    def test_temperature_below_zero_allowed(self):
        """Test that temperature below 0 is accepted (some models allow it)."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test", temperature=-0.5)
        assert config.temperature == -0.5

    def test_temperature_above_two_allowed(self):
        """Test that temperature above 2.0 is accepted."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test", temperature=2.5)
        assert config.temperature == 2.5

    def test_top_p_zero_allowed(self):
        """Test that top_p of 0 is accepted."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test", top_p=0.0)
        assert config.top_p == 0.0

    def test_top_p_one_allowed(self):
        """Test that top_p of 1.0 is accepted."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test", top_p=1.0)
        assert config.top_p == 1.0

    def test_max_tokens_zero(self):
        """Test that max_tokens of 0 is accepted."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test", max_tokens=0)
        assert config.max_tokens == 0

    def test_max_tokens_very_large(self):
        """Test that very large max_tokens is accepted."""
        config = LLMConfig(
            provider="anthropic", model="claude-3", api_key="test", max_tokens=1_000_000
        )
        assert config.max_tokens == 1_000_000

    def test_negative_max_tokens_allowed(self):
        """Test that negative max_tokens is accepted (validation deferred to provider)."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test", max_tokens=-1)
        assert config.max_tokens == -1

    def test_cache_ttl_zero(self):
        """Test that cache TTL of 0 is accepted."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test", cache_ttl=0)
        assert config.cache_ttl == 0

    def test_cache_ttl_negative_allowed(self):
        """Test that negative cache TTL is accepted."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test", cache_ttl=-1)
        assert config.cache_ttl == -1

    def test_retry_attempts_zero(self):
        """Test that retry attempts of 0 is accepted."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test", retry_attempts=0)
        assert config.retry_attempts == 0

    def test_retry_attempts_very_high(self):
        """Test that very high retry attempts is accepted."""
        config = LLMConfig(
            provider="anthropic", model="claude-3", api_key="test", retry_attempts=1000
        )
        assert config.retry_attempts == 1000


class TestMessageValidation:
    """Test message validation and edge cases."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_empty_message(self, mock_chat):
        """Test sending an empty message."""
        mock_chat.return_value = ChatResponse(
            content="Response",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3"),
        )
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        response = client.chat("")
        assert response.content == "Response"

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_very_long_message(self, mock_chat):
        """Test sending a very long message."""
        mock_chat.return_value = ChatResponse(
            content="OK",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(10000, 1, 10001, 1.0, "anthropic", "claude-3"),
        )
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        long_message = "x" * 100_000
        response = client.chat(long_message)
        assert response.content == "OK"

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_unicode_message(self, mock_chat):
        """Test sending unicode characters."""
        mock_chat.return_value = ChatResponse(
            content="OK",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(10, 1, 11, 0.0, "anthropic", "claude-3"),
        )
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        response = client.chat("你好世界 🌍 مرحبا")
        assert response.content == "OK"

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_message_with_special_characters(self, mock_chat):
        """Test message with special characters."""
        mock_chat.return_value = ChatResponse(
            content="OK",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3"),
        )
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        response = client.chat("Test\n\t\r\x00")
        assert response.content == "OK"


class TestNetworkAndTimeout:
    """Test network failure scenarios."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_connection_timeout(self, mock_chat):
        """Test handling connection timeout."""
        from socket import timeout

        mock_chat.side_effect = timeout("Connection timed out")

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        with pytest.raises(timeout):
            client.chat("Hello")

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_connection_refused(self, mock_chat):
        """Test handling connection refused."""
        from socket import error as socket_error

        mock_chat.side_effect = socket_error("Connection refused")

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        with pytest.raises(socket_error):
            client.chat("Hello")

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_dns_resolution_failure(self, mock_chat):
        """Test handling DNS resolution failure."""
        mock_chat.side_effect = Exception("Name or service not known")

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        with pytest.raises(Exception):
            client.chat("Hello")


class TestMalformedResponses:
    """Test handling of malformed responses."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_response_with_none_content(self, mock_chat):
        """Test handling response with None content."""
        mock_chat.return_value = ChatResponse(
            content=None,
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3"),
        )
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        response = client.chat("Hello")
        assert response.content is None

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_response_with_empty_content(self, mock_chat):
        """Test handling response with empty content."""
        mock_chat.return_value = ChatResponse(
            content="",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3"),
        )
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        response = client.chat("Hello")
        assert response.content == ""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_response_with_missing_usage(self, mock_chat):
        """Test handling response with None usage."""
        mock_chat.return_value = ChatResponse(
            content="Hello", provider="anthropic", model="claude-3", usage=None
        )
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        response = client.chat("Hello")
        assert response.usage is None


class TestErrorScenarios:
    """Test various error scenarios."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_rate_limit_error(self, mock_chat):
        """Test handling rate limit error."""
        mock_chat.side_effect = RateLimitError("Rate limit exceeded")
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        with pytest.raises(RateLimitError):
            client.chat("Hello")

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_authentication_error(self, mock_chat):
        """Test handling authentication error."""
        mock_chat.side_effect = AuthenticationError("Invalid API key")
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        with pytest.raises(AuthenticationError):
            client.chat("Hello")

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_provider_error(self, mock_chat):
        """Test handling generic provider error."""
        mock_chat.side_effect = ProviderError("Provider error")
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        with pytest.raises(ProviderError):
            client.chat("Hello")

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_generic_exception(self, mock_chat):
        """Test handling generic exception."""
        mock_chat.side_effect = RuntimeError("Unknown error")
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        with pytest.raises(RuntimeError):
            client.chat("Hello")


class TestTokenUsageEdgeCases:
    """Test edge cases in token usage handling."""

    def test_token_usage_with_zero_tokens(self):
        """Test token usage with zero tokens."""
        usage = TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            provider="test",
            model="test",
        )
        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.0

    def test_token_usage_with_negative_tokens(self):
        """Test token usage with negative tokens (edge case)."""
        usage = TokenUsage(
            input_tokens=-10,
            output_tokens=-20,
            total_tokens=-30,
            cost_usd=-0.001,
            provider="test",
            model="test",
        )
        assert usage.input_tokens == -10
        assert usage.cost_usd == -0.001

    def test_token_usage_with_very_large_numbers(self):
        """Test token usage with very large numbers."""
        usage = TokenUsage(
            input_tokens=10_000_000,
            output_tokens=5_000_000,
            total_tokens=15_000_000,
            cost_usd=150.0,
            provider="test",
            model="test",
        )
        assert usage.total_tokens == 15_000_000
        assert usage.cost_usd == 150.0


class TestCacheEdgeCases:
    """Test cache functionality edge cases."""

    def test_cache_with_zero_ttl(self):
        """Test cache with zero TTL (should cache but expire immediately)."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3",
            api_key="test",
            cache_responses=True,
            cache_ttl=0,
        )
        client = LLMClient(config=config)
        assert client._cache is not None

    def test_cache_disabled_but_ttl_set(self):
        """Test that TTL is ignored when cache is disabled."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3",
            api_key="test",
            cache_responses=False,
            cache_ttl=3600,
        )
        client = LLMClient(config=config)
        assert client._cache is None

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_cache_key_collision_detection(self, mock_chat):
        """Test that same message always produces same cache key."""
        mock_chat.return_value = ChatResponse(
            content="Response",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3"),
        )
        config = LLMConfig(
            provider="anthropic",
            model="claude-3",
            api_key="test",
            cache_responses=True,
            cache_ttl=3600,
        )
        client = LLMClient(config=config)

        key1 = client._get_cache_key("test")
        key2 = client._get_cache_key("test")
        assert key1 == key2


class TestProviderSelection:
    """Test provider selection and error handling."""

    def test_invalid_provider_name_raises_error(self):
        """Test that invalid provider name raises error when provider is accessed."""
        client = LLMClient(provider="nonexistent", model="test")
        with pytest.raises(ProviderError):
            _ = client.provider  # Error raised on first access

    def test_provider_name_case_sensitivity(self):
        """Test provider selection with different cases."""
        # Should be case-insensitive due to .lower() in _create_provider
        client = LLMClient(provider="ANTHROPIC", model="claude-3", api_key="test")
        assert client.config.provider == "ANTHROPIC"


class TestConcurrentRequests:
    """Test handling of concurrent requests (thread safety)."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_multiple_sequential_requests(self, mock_chat):
        """Test multiple sequential requests."""
        responses = [
            ChatResponse(
                content=f"Response {i}",
                provider="anthropic",
                model="claude-3",
                usage=TokenUsage(1, 1, 2, 0.0, "anthropic", "claude-3"),
            )
            for i in range(5)
        ]
        mock_chat.side_effect = responses

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        results = [client.chat(f"Message {i}") for i in range(5)]

        assert len(results) == 5
        assert all(isinstance(r, ChatResponse) for r in results)

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_request_after_error(self, mock_chat):
        """Test that client recovers after error."""
        mock_chat.side_effect = [
            Exception("Error"),
            ChatResponse(
                content="Success",
                provider="anthropic",
                model="claude-3",
                usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3"),
            ),
        ]

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")

        with pytest.raises(Exception):
            client.chat("First request")

        response = client.chat("Second request")
        assert response.content == "Success"


class TestParameterEdgeCases:
    """Test edge cases in parameter passing."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_chat_with_all_parameters(self, mock_chat):
        """Test chat with all possible parameters."""
        mock_chat.return_value = ChatResponse(
            content="OK",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3"),
        )
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")

        response = client.chat(
            "Message", temperature=0.5, top_p=0.9, max_tokens=100, custom_param="custom_value"
        )
        assert response.content == "OK"

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_chat_with_no_optional_parameters(self, mock_chat):
        """Test chat with no optional parameters."""
        mock_chat.return_value = ChatResponse(
            content="OK",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3"),
        )
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        response = client.chat("Message")
        assert response.content == "OK"
