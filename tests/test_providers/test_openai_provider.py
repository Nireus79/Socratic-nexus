"""Tests for OpenAI GPT provider."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from socratic_nexus.models import LLMConfig, ChatResponse, TokenUsage
from socratic_nexus.providers.openai import OpenAIProvider
from socratic_nexus.exceptions import (
    RateLimitError,
    AuthenticationError,
    ProviderError,
)


class TestOpenAIInitialization:
    """Test OpenAI provider initialization."""

    def test_initialization_with_api_key(self):
        """Test provider initializes with API key."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test-key")
        provider = OpenAIProvider(config)
        assert provider.config == config
        assert provider._client is None

    def test_initialization_without_api_key_raises_error(self):
        """Test initialization fails without API key."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key=None)
        with pytest.raises(AuthenticationError):
            OpenAIProvider(config)

    def test_initialization_with_empty_api_key_raises_error(self):
        """Test initialization fails with empty API key."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="")
        with pytest.raises(AuthenticationError):
            OpenAIProvider(config)


class TestOpenAIClientInitialization:
    """Test lazy initialization of OpenAI clients."""

    @patch("openai.OpenAI")
    def test_client_property_initializes_on_first_access(self, mock_openai_class):
        """Test client is created on first access."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)

        assert provider._client is None
        client = provider.client
        assert client == mock_client
        mock_openai_class.assert_called_once_with(api_key="sk-test")

    @patch("openai.OpenAI")
    def test_client_property_missing_openai_package(self, mock_openai_class):
        """Test error when openai package is not installed."""
        mock_openai_class.side_effect = ImportError("No module named 'openai'")

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)

        with pytest.raises(ProviderError) as exc_info:
            _ = provider.client
        assert "openai package is required" in str(exc_info.value)


class TestOpenAIChat:
    """Test OpenAI chat method."""

    @patch("openai.OpenAI")
    def test_chat_basic_message(self, mock_openai_class):
        """Test basic chat message."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello OpenAI!"), finish_reason="stop")]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
        mock_response.id = "chatcmpl-123"
        mock_response.model = "gpt-4"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)

        response = provider.chat("Hello!")

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello OpenAI!"
        assert response.provider == "openai"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert response.usage.total_tokens == 30

    @patch("openai.OpenAI")
    def test_chat_with_custom_parameters(self, mock_openai_class):
        """Test chat with custom parameters."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"), finish_reason="stop")]
        mock_response.usage = Mock(prompt_tokens=5, completion_tokens=10)
        mock_response.id = "chatcmpl-123"
        mock_response.model = "gpt-4"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)

        provider.chat("Message", temperature=0.2, max_tokens=512)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 512

    @patch("openai.OpenAI")
    def test_chat_invokes_usage_callback(self, mock_openai_class):
        """Test chat invokes usage callbacks."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"), finish_reason="stop")]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_response.id = "chatcmpl-123"
        mock_response.model = "gpt-4"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)

        callback = Mock()
        provider.add_usage_callback(callback)

        provider.chat("Message")

        callback.assert_called_once()
        usage_arg = callback.call_args[0][0]
        assert isinstance(usage_arg, TokenUsage)
        assert usage_arg.input_tokens == 100
        assert usage_arg.output_tokens == 50

    @patch("openai.OpenAI")
    def test_chat_rate_limit_error(self, mock_openai_class):
        """Test chat handles rate limit errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("429 rate_limit_exceeded")
        mock_openai_class.return_value = mock_client

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)

        with pytest.raises(RateLimitError):
            provider.chat("Message")

    @patch("openai.OpenAI")
    def test_chat_invalid_api_key_error(self, mock_openai_class):
        """Test chat handles invalid API key error."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("invalid_api_key")
        mock_openai_class.return_value = mock_client

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-invalid")
        provider = OpenAIProvider(config)

        with pytest.raises(AuthenticationError):
            provider.chat("Message")


class TestOpenAIAsync:
    """Test OpenAI async methods."""

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_achat_basic_message(self, mock_async_class):
        """Test basic async chat message."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello async!"), finish_reason="stop")]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
        mock_response.id = "chatcmpl-123"
        mock_response.model = "gpt-4"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_class.return_value = mock_client

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)

        response = await provider.achat("Hello!")

        assert response.content == "Hello async!"
        assert response.usage.input_tokens == 10


class TestOpenAIStreaming:
    """Test OpenAI streaming."""

    @patch("openai.OpenAI")
    def test_stream_basic(self, mock_openai_class):
        """Test basic streaming."""
        from unittest.mock import MagicMock

        # Create mock stream chunks
        chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))], usage=None),
            Mock(choices=[Mock(delta=Mock(content=" "))], usage=None),
            Mock(choices=[Mock(delta=Mock(content="stream"))], usage=None),
            Mock(
                choices=[Mock(delta=Mock(content="!"))],
                finish_reason="stop",
                usage=Mock(prompt_tokens=10, completion_tokens=20),
            ),
        ]

        # Create a proper context manager for the stream
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = iter(chunks)
        mock_stream.__exit__.return_value = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_stream
        mock_openai_class.return_value = mock_client

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)

        callback_chunks = []

        def on_chunk(chunk):
            callback_chunks.append(chunk)

        response = provider.stream("Hello!", on_chunk=on_chunk)

        assert response.usage.input_tokens == 10
        assert response.content == "Hello stream!"


class TestOpenAIErrorHandling:
    """Test OpenAI error handling."""

    @patch("openai.OpenAI")
    def test_handle_openai_error_generic(self, mock_openai_class):
        """Test generic error handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Unknown error")
        mock_openai_class.return_value = mock_client

        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)

        with pytest.raises(ProviderError) as exc_info:
            provider.chat("Message")

        assert "OPENAI_ERROR" in str(exc_info.value.error_code)


class TestOpenAIRepresentation:
    """Test OpenAI provider representation."""

    def test_repr(self):
        """Test provider representation."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="sk-test")
        provider = OpenAIProvider(config)
        repr_str = repr(provider)

        assert "OpenAIProvider" in repr_str
        assert "gpt-4" in repr_str
