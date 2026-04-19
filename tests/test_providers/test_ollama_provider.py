"""Tests for Ollama provider."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from socratic_nexus.models import LLMConfig, ChatResponse, TokenUsage
from socratic_nexus.providers.ollama import OllamaProvider
from socratic_nexus.exceptions import (
    ProviderError,
)


class TestOllamaInitialization:
    """Test Ollama provider initialization."""

    def test_initialization_with_base_url(self):
        """Test provider initializes with base URL."""
        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)
        assert provider.config == config

    def test_initialization_without_base_url_uses_default(self):
        """Test initialization uses default base URL."""
        config = LLMConfig(provider="ollama", model="llama2")
        provider = OllamaProvider(config)
        assert provider.config == config
        # Default is usually localhost:11434

    def test_initialization_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        config = LLMConfig(provider="ollama", model="llama2", base_url="http://192.168.1.100:11434")
        provider = OllamaProvider(config)
        assert provider.config.base_url == "http://192.168.1.100:11434"


class TestOllamaClientInitialization:
    """Test Ollama client initialization."""

    @patch("ollama.AsyncClient")
    @patch("ollama.Client")
    def test_client_property_initializes_on_first_access(
        self, mock_client_class, mock_async_client_class
    ):
        """Test client is created on first access."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        assert provider._client is None
        client = provider.client
        assert client == mock_client
        mock_client_class.assert_called_once_with(host="http://localhost:11434")

    @patch("ollama.AsyncClient")
    @patch("ollama.Client")
    def test_async_client_property_initializes_on_first_access(
        self, mock_client_class, mock_async_client_class
    ):
        """Test async client is created on first access."""
        mock_async_client = AsyncMock()
        mock_async_client_class.return_value = mock_async_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        assert provider._async_client is None
        async_client = provider.async_client
        assert async_client == mock_async_client
        mock_async_client_class.assert_called_once_with(host="http://localhost:11434")


class TestOllamaChat:
    """Test Ollama chat method."""

    @patch("ollama.Client")
    def test_chat_basic_message(self, mock_client_class):
        """Test basic chat message."""
        mock_response = {
            "message": {"content": "Hello Ollama!"},
            "prompt_eval_count": 10,
            "eval_count": 20,
            "done": True,
        }

        mock_client = Mock()
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        response = provider.chat("Hello!")

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello Ollama!"
        assert response.provider == "ollama"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20

    @patch("ollama.Client")
    def test_chat_with_custom_temperature(self, mock_client_class):
        """Test chat with custom temperature."""
        mock_response = {
            "message": {"content": "Response"},
            "prompt_eval_count": 5,
            "eval_count": 10,
            "done": True,
        }

        mock_client = Mock()
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        provider.chat("Message", temperature=0.2)

        call_kwargs = mock_client.chat.call_args[1]
        assert "options" in call_kwargs
        assert call_kwargs["options"]["temperature"] == 0.2

    @patch("ollama.Client")
    def test_chat_invokes_usage_callback(self, mock_client_class):
        """Test chat invokes usage callbacks."""
        mock_response = {
            "message": {"content": "Response"},
            "prompt_eval_count": 100,
            "eval_count": 50,
            "done": True,
        }

        mock_client = Mock()
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        callback = Mock()
        provider.add_usage_callback(callback)

        provider.chat("Message")

        callback.assert_called_once()
        usage_arg = callback.call_args[0][0]
        assert isinstance(usage_arg, TokenUsage)
        assert usage_arg.input_tokens == 100

    @patch("ollama.Client")
    def test_chat_with_top_p(self, mock_client_class):
        """Test chat with top_p parameter."""
        mock_response = {
            "message": {"content": "Response"},
            "prompt_eval_count": 5,
            "eval_count": 10,
            "done": True,
        }

        mock_client = Mock()
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        provider.chat("Message", top_p=0.8)

        call_kwargs = mock_client.chat.call_args[1]
        assert "options" in call_kwargs
        assert call_kwargs["options"]["top_p"] == 0.8


class TestOllamaAsync:
    """Test Ollama async methods."""

    @pytest.mark.asyncio
    @patch("ollama.AsyncClient")
    async def test_achat_basic_message(self, mock_async_client_class):
        """Test basic async chat message."""
        mock_response = {
            "message": {"content": "Hello async!"},
            "prompt_eval_count": 10,
            "eval_count": 20,
            "done": True,
        }

        mock_async_client = AsyncMock()
        mock_async_client.chat = AsyncMock(return_value=mock_response)
        mock_async_client_class.return_value = mock_async_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        response = await provider.achat("Hello!")

        assert response.content == "Hello async!"
        assert response.provider == "ollama"
        assert response.usage.input_tokens == 10


class TestOllamaStreaming:
    """Test Ollama streaming."""

    @patch("ollama.Client")
    def test_stream_basic(self, mock_client_class):
        """Test basic streaming."""
        # Create mock stream chunks (dictionary format)
        chunks = [
            {"message": {"content": "Hello"}, "done": False},
            {"message": {"content": " "}, "done": False},
            {"message": {"content": "stream"}, "done": False},
            {"message": {"content": "!"}, "done": True, "prompt_eval_count": 10, "eval_count": 20},
        ]

        mock_client = Mock()
        mock_client.chat.return_value = iter(chunks)
        mock_client_class.return_value = mock_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        callback_chunks = []

        def on_chunk(chunk):
            callback_chunks.append(chunk)

        response = provider.stream("Hello!", on_chunk=on_chunk)

        assert response.content == "Hello stream!"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20
        assert len(callback_chunks) == 4


class TestOllamaErrorHandling:
    """Test Ollama error handling."""

    @patch("ollama.Client")
    def test_handle_ollama_error_connection_refused(self, mock_client_class):
        """Test connection refused error handling."""
        mock_client = Mock()
        mock_client.chat.side_effect = Exception("Connection refused")
        mock_client_class.return_value = mock_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        with pytest.raises(ProviderError):
            provider.chat("Message")

    @patch("ollama.Client")
    def test_handle_ollama_error_generic(self, mock_client_class):
        """Test generic error handling."""
        mock_client = Mock()
        mock_client.chat.side_effect = Exception("Unknown error occurred")
        mock_client_class.return_value = mock_client

        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)

        with pytest.raises(ProviderError):
            provider.chat("Message")


class TestOllamaRepresentation:
    """Test Ollama provider representation."""

    def test_repr(self):
        """Test provider representation."""
        config = LLMConfig(provider="ollama", model="llama2", base_url="http://localhost:11434")
        provider = OllamaProvider(config)
        repr_str = repr(provider)

        assert "OllamaProvider" in repr_str
        assert "llama2" in repr_str
