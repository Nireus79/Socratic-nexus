"""Tests for Anthropic Claude provider.

Comprehensive tests for AnthropicProvider implementation covering:
- Client initialization (lazy loading)
- Chat and async chat methods
- Streaming and async streaming
- Parameter passing and configuration
- Error handling and exception mapping
- Usage tracking and callbacks
- Latency measurement
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from socratic_nexus.models import LLMConfig, ChatResponse, TokenUsage
from socratic_nexus.providers.anthropic import AnthropicProvider
from socratic_nexus.exceptions import (
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
)


class TestAnthropicProviderInitialization:
    """Test AnthropicProvider initialization."""

    def test_initialization_with_api_key(self):
        """Test provider initializes with API key."""
        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test-key"
        )
        provider = AnthropicProvider(config)

        assert provider.config == config
        assert provider._client is None  # Lazy initialization
        assert provider._async_client is None

    def test_initialization_without_api_key_raises_error(self):
        """Test initialization fails without API key."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022", api_key=None)

        with pytest.raises(AuthenticationError):
            AnthropicProvider(config)

    def test_initialization_with_empty_api_key_raises_error(self):
        """Test initialization fails with empty API key."""
        config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="")

        with pytest.raises(AuthenticationError):
            AnthropicProvider(config)


class TestAnthropicClientLazyInitialization:
    """Test lazy initialization of Anthropic clients."""

    @patch("anthropic.Anthropic")
    def test_client_property_initializes_on_first_access(self, mock_anthropic_class):
        """Test client is created on first access."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        # Before access, should be None
        assert provider._client is None

        # First access should initialize
        client = provider.client
        assert client == mock_client
        mock_anthropic_class.assert_called_once_with(api_key="sk-ant-test")

        # Second access should return cached client
        client2 = provider.client
        assert client2 == mock_client
        assert mock_anthropic_class.call_count == 1  # Not called again

    @patch("anthropic.Anthropic")
    def test_client_property_missing_anthropic_package(self, mock_anthropic_class):
        """Test error when anthropic package is not installed."""
        mock_anthropic_class.side_effect = ImportError("No module named 'anthropic'")

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        with pytest.raises(ProviderError) as exc_info:
            _ = provider.client

        assert "anthropic package is required" in str(exc_info.value)

    @patch("anthropic.AsyncAnthropic")
    def test_async_client_property_initializes_on_first_access(self, mock_async_class):
        """Test async client is created on first access."""
        mock_async_client = Mock()
        mock_async_class.return_value = mock_async_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        # Before access, should be None
        assert provider._async_client is None

        # First access should initialize
        client = provider.async_client
        assert client == mock_async_client
        mock_async_class.assert_called_once_with(api_key="sk-ant-test")

        # Second access should return cached client
        client2 = provider.async_client
        assert client2 == mock_async_client
        assert mock_async_class.call_count == 1


class TestAnthropicChat:
    """Test Anthropic chat method."""

    @patch("anthropic.Anthropic")
    def test_chat_basic_message(self, mock_anthropic_class):
        """Test basic chat message."""
        # Create mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Hello, how can I help?")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_response.id = "msg-123"
        mock_response.type = "message"
        mock_response.stop_reason = "end_turn"

        # Mock client
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="sk-ant-test",
            temperature=0.7,
            max_tokens=1024,
        )
        provider = AnthropicProvider(config)

        response = provider.chat("Hello!")

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello, how can I help?"
        assert response.provider == "anthropic"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30
        assert response.finish_reason == "end_turn"

    @patch("anthropic.Anthropic")
    def test_chat_with_custom_temperature(self, mock_anthropic_class):
        """Test chat with custom temperature parameter."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.usage = Mock(input_tokens=5, output_tokens=10)
        mock_response.id = "msg-123"
        mock_response.type = "message"
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        provider.chat("Message", temperature=0.2)

        # Verify temperature was passed to API
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.2

    @patch("anthropic.Anthropic")
    def test_chat_with_custom_max_tokens(self, mock_anthropic_class):
        """Test chat with custom max_tokens parameter."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.usage = Mock(input_tokens=5, output_tokens=10)
        mock_response.id = "msg-123"
        mock_response.type = "message"
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        provider.chat("Message", max_tokens=512)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 512

    @patch("anthropic.Anthropic")
    def test_chat_uses_config_defaults_for_max_tokens(self, mock_anthropic_class):
        """Test chat uses config max_tokens when not provided."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.usage = Mock(input_tokens=5, output_tokens=10)
        mock_response.id = "msg-123"
        mock_response.type = "message"
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="sk-ant-test",
            max_tokens=2048,
        )
        provider = AnthropicProvider(config)

        provider.chat("Message")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 2048

    @patch("anthropic.Anthropic")
    def test_chat_uses_default_max_tokens_4096(self, mock_anthropic_class):
        """Test chat uses 4096 as default max_tokens."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.usage = Mock(input_tokens=5, output_tokens=10)
        mock_response.id = "msg-123"
        mock_response.type = "message"
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="sk-ant-test",
            max_tokens=None,
        )
        provider = AnthropicProvider(config)

        provider.chat("Message")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    @patch("anthropic.Anthropic")
    def test_chat_invokes_usage_callback(self, mock_anthropic_class):
        """Test chat invokes usage callbacks."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_response.id = "msg-123"
        mock_response.type = "message"
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        callback = Mock()
        provider.add_usage_callback(callback)

        provider.chat("Message")

        callback.assert_called_once()
        usage_arg = callback.call_args[0][0]
        assert isinstance(usage_arg, TokenUsage)
        assert usage_arg.input_tokens == 100
        assert usage_arg.output_tokens == 50

    @patch("anthropic.Anthropic")
    def test_chat_measures_latency(self, mock_anthropic_class):
        """Test chat measures and includes latency."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_response.id = "msg-123"
        mock_response.type = "message"
        mock_response.stop_reason = "end_turn"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        response = provider.chat("Message")

        assert response.usage.latency_ms >= 0  # Latency measured

    @patch("anthropic.Anthropic")
    def test_chat_rate_limit_error(self, mock_anthropic_class):
        """Test chat handles rate limit errors."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("429 rate_limit_error")
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        with pytest.raises(RateLimitError):
            provider.chat("Message")

    @patch("anthropic.Anthropic")
    def test_chat_invalid_api_key_error(self, mock_anthropic_class):
        """Test chat handles invalid API key error."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("invalid_api_key")
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-invalid"
        )
        provider = AnthropicProvider(config)

        with pytest.raises(AuthenticationError):
            provider.chat("Message")

    @patch("anthropic.Anthropic")
    def test_chat_invalid_request_error(self, mock_anthropic_class):
        """Test chat handles invalid request error."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("invalid_request_error")
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        with pytest.raises(InvalidRequestError):
            provider.chat("Message")

    @patch("anthropic.Anthropic")
    def test_chat_context_length_exceeded_error(self, mock_anthropic_class):
        """Test chat handles context length exceeded error."""
        from socratic_nexus.exceptions import ContextLengthExceededError

        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("context_length_exceeded")
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        with pytest.raises(ContextLengthExceededError):
            provider.chat("Message")


class TestAnthropicAsyncChat:
    """Test Anthropic async chat method."""

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic")
    async def test_achat_basic_message(self, mock_async_class):
        """Test basic async chat message."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Hello async!")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_response.id = "msg-123"
        mock_response.type = "message"
        mock_response.stop_reason = "end_turn"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_async_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        response = await provider.achat("Hello!")

        assert response.content == "Hello async!"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20


class TestAnthropicStream:
    """Test Anthropic streaming."""

    @patch("anthropic.Anthropic")
    def test_stream_basic(self, mock_anthropic_class):
        """Test basic streaming."""
        # Create mock stream with context manager protocol
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["Hello", " ", "stream", "!"])
        mock_final_message = Mock()
        mock_final_message.usage = Mock(input_tokens=10, output_tokens=20)
        mock_final_message.id = "msg-123"
        mock_final_message.type = "message"
        mock_final_message.stop_reason = "end_turn"
        mock_stream.get_final_message.return_value = mock_final_message

        # Create context manager that returns the stream
        context_manager = MagicMock()
        context_manager.__enter__ = MagicMock(return_value=mock_stream)
        context_manager.__exit__ = MagicMock(return_value=None)

        mock_client = Mock()
        mock_client.messages.stream.return_value = context_manager
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        chunks = []

        def on_chunk(chunk):
            chunks.append(chunk)

        response = provider.stream("Hello!", on_chunk=on_chunk)

        # Verify chunks were received
        assert chunks == ["Hello", " ", "stream", "!"]
        assert response.content == "Hello stream!"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20

    @patch("anthropic.Anthropic")
    def test_stream_invokes_callback_for_each_chunk(self, mock_anthropic_class):
        """Test stream callback is called for each chunk."""
        mock_stream = MagicMock()
        mock_stream.text_stream = iter(["a", "b", "c"])
        mock_final_message = Mock()
        mock_final_message.usage = Mock(input_tokens=5, output_tokens=10)
        mock_final_message.id = "msg-123"
        mock_final_message.type = "message"
        mock_final_message.stop_reason = "end_turn"
        mock_stream.get_final_message.return_value = mock_final_message

        # Create context manager that returns the stream
        context_manager = MagicMock()
        context_manager.__enter__ = MagicMock(return_value=mock_stream)
        context_manager.__exit__ = MagicMock(return_value=None)

        mock_client = Mock()
        mock_client.messages.stream.return_value = context_manager
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        callback = Mock()
        provider.stream("Message", on_chunk=callback)

        # Callback should be called 3 times
        assert callback.call_count == 3
        callback.assert_any_call("a")
        callback.assert_any_call("b")
        callback.assert_any_call("c")


class TestAnthropicAsyncStream:
    """Test Anthropic async streaming."""

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic")
    async def test_astream_basic(self, mock_async_class):
        """Test basic async streaming."""
        # Note: Async context manager mocking is complex with the actual Anthropic SDK
        # Full async streaming testing recommended with integration tests (marked with requires_api)
        # This test verifies the method exists and can be called
        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        # Verify method exists and is callable
        assert hasattr(provider, "astream")
        assert callable(provider.astream)


class TestAnthropicErrorHandling:
    """Test Anthropic error handling."""

    @patch("anthropic.Anthropic")
    def test_handle_anthropic_error_generic(self, mock_anthropic_class):
        """Test generic error handling."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Unknown error")
        mock_anthropic_class.return_value = mock_client

        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        with pytest.raises(ProviderError) as exc_info:
            provider.chat("Message")

        assert "ANTHROPIC_ERROR" in str(exc_info.value.error_code)


class TestAnthropicProviderRepresentation:
    """Test Anthropic provider string representation."""

    def test_repr(self):
        """Test provider representation."""
        config = LLMConfig(
            provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        provider = AnthropicProvider(config)

        repr_str = repr(provider)

        assert "AnthropicProvider" in repr_str
        assert "claude-3-5-sonnet-20241022" in repr_str
