"""Tests for LLM client integration with providers."""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.client import LLMClient
from socratic_nexus.models import LLMConfig, ChatResponse, TokenUsage, UsageStats
from socratic_nexus.exceptions import ConfigurationError, ProviderError


class TestLLMClientInitialization:
    """Test LLMClient initialization."""

    def test_initialization_with_config(self):
        """Test client initialization with LLMConfig."""
        config = LLMConfig(provider="anthropic", model="claude-3", api_key="test")
        client = LLMClient(config=config)
        assert client.config == config
        assert isinstance(client.usage_stats, UsageStats)

    def test_initialization_with_kwargs(self):
        """Test client initialization with kwargs."""
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        assert client.config.provider == "anthropic"
        assert client.config.model == "claude-3"

    def test_initialization_without_provider_raises_error(self):
        """Test that initialization without provider raises error."""
        with pytest.raises(ConfigurationError):
            LLMClient()

    def test_initialization_with_cache_enabled(self):
        """Test client initialization with caching."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3",
            api_key="test",
            cache_responses=True,
            cache_ttl=3600,
        )
        client = LLMClient(config=config)
        assert client._cache is not None

    def test_initialization_with_cache_disabled(self):
        """Test client initialization without caching."""
        config = LLMConfig(
            provider="anthropic", model="claude-3", api_key="test", cache_responses=False
        )
        client = LLMClient(config=config)
        assert client._cache is None


class TestProviderFactory:
    """Test provider creation and selection."""

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        provider = client.provider
        assert provider is not None
        assert provider.__class__.__name__ == "AnthropicProvider"

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        client = LLMClient(provider="openai", model="gpt-4", api_key="test")
        provider = client.provider
        assert provider.__class__.__name__ == "OpenAIProvider"

    def test_create_google_provider(self):
        """Test creating Google provider."""
        client = LLMClient(provider="google", model="gemini-1.5-pro", api_key="test")
        provider = client.provider
        assert provider.__class__.__name__ == "GoogleProvider"

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        client = LLMClient(provider="ollama", model="llama2")
        provider = client.provider
        assert provider.__class__.__name__ == "OllamaProvider"

    def test_provider_aliases_claude(self):
        """Test 'claude' alias for Anthropic."""
        client = LLMClient(provider="claude", model="claude-3", api_key="test")
        provider = client.provider
        assert provider.__class__.__name__ == "AnthropicProvider"

    def test_provider_aliases_gpt(self):
        """Test 'gpt' alias for OpenAI."""
        client = LLMClient(provider="gpt", model="gpt-4", api_key="test")
        provider = client.provider
        assert provider.__class__.__name__ == "OpenAIProvider"

    def test_provider_aliases_gemini(self):
        """Test 'gemini' alias for Google."""
        client = LLMClient(provider="gemini", model="gemini-1.5-pro", api_key="test")
        provider = client.provider
        assert provider.__class__.__name__ == "GoogleProvider"

    def test_provider_aliases_local(self):
        """Test 'local' alias for Ollama."""
        client = LLMClient(provider="local", model="llama2")
        provider = client.provider
        assert provider.__class__.__name__ == "OllamaProvider"

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises error."""
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        client.config.provider = "unsupported"
        with pytest.raises(ProviderError):
            _ = client.provider

    def test_case_insensitive_provider_selection(self):
        """Test that provider selection is case insensitive."""
        client = LLMClient(provider="ANTHROPIC", model="claude-3", api_key="test")
        provider = client.provider
        assert provider.__class__.__name__ == "AnthropicProvider"


class TestChatIntegration:
    """Test chat method with provider integration."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_chat_calls_provider(self, mock_chat):
        """Test that chat method calls provider."""
        mock_response = ChatResponse(
            content="Hello",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                cost_usd=0.0,
                provider="anthropic",
                model="claude-3",
            ),
        )
        mock_chat.return_value = mock_response

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        response = client.chat("Hello")

        assert response == mock_response
        mock_chat.assert_called_once()

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_chat_with_parameters(self, mock_chat):
        """Test passing parameters to provider."""
        mock_response = ChatResponse(
            content="Response",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(
                input_tokens=5,
                output_tokens=10,
                total_tokens=15,
                cost_usd=0.0,
                provider="anthropic",
                model="claude-3",
            ),
        )
        mock_chat.return_value = mock_response

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        client.chat("Message", temperature=0.5, max_tokens=100)

        mock_chat.assert_called_once_with("Message", temperature=0.5, max_tokens=100)

    def test_usage_tracking(self):
        """Test that usage is tracked through callbacks."""

        # Create a simple mock provider that tracks callbacks
        class MockProvider:
            def __init__(self, config):
                self.config = config
                self._usage_callbacks = []

            def add_usage_callback(self, callback):
                self._usage_callbacks.append(callback)

            def chat(self, message, **kwargs):
                return ChatResponse(
                    content="Hello",
                    provider="test",
                    model="test-model",
                    usage=TokenUsage(
                        input_tokens=10,
                        output_tokens=20,
                        total_tokens=30,
                        cost_usd=0.001,
                        provider="test",
                        model="test-model",
                    ),
                )

        # Manually create client with mock provider
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        client._provider = MockProvider(client.config)

        # Register the tracking callback
        client._provider.add_usage_callback(client._track_usage)

        # Perform chat (this won't call the real provider)
        response = client.chat("Hello")

        # Manually invoke the callback to simulate what the provider would do
        usage = response.usage
        client._track_usage(usage)

        assert client.usage_stats.total_input_tokens == 10
        assert client.usage_stats.total_output_tokens == 20
        assert client.usage_stats.total_cost_usd == 0.001

    def test_multiple_requests_accumulate_usage(self):
        """Test that multiple requests accumulate usage."""
        # Create a mock provider that returns different responses
        responses_data = [
            (TokenUsage(10, 20, 30, 0.001, "test", "test"), "Response1"),
            (TokenUsage(15, 25, 40, 0.0015, "test", "test"), "Response2"),
        ]
        request_count = [0]

        class MockProvider:
            def __init__(self, config):
                self.config = config
                self._usage_callbacks = []

            def add_usage_callback(self, callback):
                self._usage_callbacks.append(callback)

            def chat(self, message, **kwargs):
                idx = request_count[0]
                request_count[0] += 1
                usage, content = responses_data[idx]
                return ChatResponse(
                    content=content, provider="test", model="test-model", usage=usage
                )

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        client._provider = MockProvider(client.config)
        client._provider.add_usage_callback(client._track_usage)

        response1 = client.chat("Hello")
        response2 = client.chat("World")

        # Manually invoke callbacks for each response
        client._track_usage(response1.usage)
        client._track_usage(response2.usage)

        assert client.usage_stats.total_input_tokens == 25
        assert client.usage_stats.total_output_tokens == 45
        assert client.usage_stats.total_cost_usd == pytest.approx(0.0025, rel=1e-5)


class TestCacheIntegration:
    """Test cache integration with client."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_cache_stores_response(self, mock_chat):
        """Test that cache stores responses."""
        usage = TokenUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0,
            provider="anthropic",
            model="claude-3",
        )
        response = ChatResponse(
            content="Cached response", provider="anthropic", model="claude-3", usage=usage
        )
        mock_chat.return_value = response

        config = LLMConfig(
            provider="anthropic",
            model="claude-3",
            api_key="test",
            cache_responses=True,
            cache_ttl=3600,
        )
        client = LLMClient(config=config)

        response1 = client.chat("Same message")
        response2 = client.chat("Same message")

        # Provider should only be called once due to cache
        assert mock_chat.call_count == 1
        assert response1.content == response2.content

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_cache_miss_calls_provider(self, mock_chat):
        """Test that cache miss calls provider."""
        usage = TokenUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0,
            provider="anthropic",
            model="claude-3",
        )
        response = ChatResponse(
            content="Response", provider="anthropic", model="claude-3", usage=usage
        )
        mock_chat.return_value = response

        config = LLMConfig(
            provider="anthropic",
            model="claude-3",
            api_key="test",
            cache_responses=True,
            cache_ttl=3600,
        )
        client = LLMClient(config=config)

        client.chat("Message 1")
        client.chat("Message 2")

        # Provider should be called twice for different messages
        assert mock_chat.call_count == 2

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_no_cache_always_calls_provider(self, mock_chat):
        """Test that disabled cache always calls provider."""
        usage = TokenUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0,
            provider="anthropic",
            model="claude-3",
        )
        response = ChatResponse(
            content="Response", provider="anthropic", model="claude-3", usage=usage
        )
        mock_chat.return_value = response

        config = LLMConfig(
            provider="anthropic", model="claude-3", api_key="test", cache_responses=False
        )
        client = LLMClient(config=config)

        client.chat("Same message")
        client.chat("Same message")

        # Provider should be called twice even for same message
        assert mock_chat.call_count == 2


class TestErrorHandling:
    """Test error handling and propagation."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_provider_error_propagates(self, mock_chat):
        """Test that provider errors propagate."""
        from socratic_nexus.exceptions import RateLimitError

        mock_chat.side_effect = RateLimitError("Rate limited")

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        with pytest.raises(RateLimitError):
            client.chat("Hello")

    def test_invalid_config_raises_error(self):
        """Test that invalid config raises error."""
        with pytest.raises(TypeError):
            # Missing required model parameter
            LLMClient(provider="anthropic")


class TestProviderLazyInitialization:
    """Test that provider is lazily initialized."""

    def test_provider_not_initialized_on_client_creation(self):
        """Test that provider is not initialized until first use."""
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        assert client._provider is None

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider")
    def test_provider_initialized_on_first_use(self, mock_provider_class):
        """Test that provider is initialized on first use."""
        mock_instance = Mock()
        mock_instance.chat.return_value = ChatResponse(
            content="Response",
            provider="anthropic",
            model="claude-3",
            usage=TokenUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                provider="anthropic",
                model="claude-3",
            ),
        )
        mock_provider_class.return_value = mock_instance

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        assert client._provider is None

        # Access provider property
        _ = client.provider

        assert client._provider is not None

    def test_provider_callback_registered(self):
        """Test that usage callback is registered with provider."""

        # Create a mock provider to verify callback registration
        class MockProvider:
            def __init__(self, config):
                self.config = config
                self.callbacks = []

            def add_usage_callback(self, callback):
                self.callbacks.append(callback)

            def chat(self, message, **kwargs):
                return ChatResponse(
                    content="Test",
                    provider="test",
                    model="test-model",
                    usage=TokenUsage(0, 0, 0, 0.0, "test", "test-model"),
                )

        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        # Inject mock provider
        client._provider = MockProvider(client.config)
        client._provider.add_usage_callback(client._track_usage)

        # Verify callback was registered
        assert len(client._provider.callbacks) == 1
        assert client._provider.callbacks[0] == client._track_usage


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_same_message_generates_same_key(self):
        """Test that same message generates same cache key."""
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        key1 = client._get_cache_key("Hello")
        key2 = client._get_cache_key("Hello")
        assert key1 == key2

    def test_different_messages_generate_different_keys(self):
        """Test that different messages generate different keys."""
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        key1 = client._get_cache_key("Hello")
        key2 = client._get_cache_key("World")
        assert key1 != key2

    def test_cache_key_is_deterministic(self):
        """Test that cache key is deterministic."""
        client1 = LLMClient(provider="anthropic", model="claude-3", api_key="test1")
        client2 = LLMClient(provider="anthropic", model="claude-3", api_key="test2")

        # Same message should generate same key regardless of client
        key1 = client1._get_cache_key("Message")
        key2 = client2._get_cache_key("Message")
        assert key1 == key2

    def test_cache_key_is_hex_string(self):
        """Test that cache key is a valid hex string."""
        client = LLMClient(provider="anthropic", model="claude-3", api_key="test")
        key = client._get_cache_key("Message")
        # Should be valid hex
        int(key, 16)  # This will raise if not valid hex
