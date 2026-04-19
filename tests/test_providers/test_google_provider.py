"""Tests for Google Gemini provider."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from socratic_nexus.models import LLMConfig, ChatResponse, TokenUsage
from socratic_nexus.providers.google import GoogleProvider
from socratic_nexus.exceptions import (
    RateLimitError,
    AuthenticationError,
)


class TestGoogleInitialization:
    """Test Google provider initialization."""

    def test_initialization_with_api_key(self):
        """Test provider initializes with API key."""
        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="test-key")
        provider = GoogleProvider(config)
        assert provider.config == config

    def test_initialization_without_api_key_raises_error(self):
        """Test initialization fails without API key."""
        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key=None)
        with pytest.raises(AuthenticationError):
            GoogleProvider(config)

    def test_initialization_with_empty_api_key_raises_error(self):
        """Test initialization fails with empty API key."""
        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="")
        with pytest.raises(AuthenticationError):
            GoogleProvider(config)


class TestGoogleClientInitialization:
    """Test Google model initialization."""

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    def test_client_property_initializes_on_first_access(self, mock_model_class, mock_configure):
        """Test model is created on first access."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="test-key")
        provider = GoogleProvider(config)

        model = provider.model
        assert model == mock_model
        mock_configure.assert_called_once_with(api_key="test-key")


class TestGoogleChat:
    """Test Google chat method."""

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_chat_basic_message(self, mock_configure, mock_model_class):
        """Test basic chat message."""
        mock_response = Mock()
        mock_response.text = "Hello Google!"
        mock_response.usage_metadata = Mock(prompt_character_count=10, candidates_token_count=20)
        mock_response.candidates = [Mock(finish_reason=Mock(name="STOP"))]

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="test-key")
        provider = GoogleProvider(config)

        response = provider.chat("Hello!")

        assert isinstance(response, ChatResponse)
        assert response.content == "Hello Google!"
        assert response.provider == "google"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_chat_with_custom_temperature(self, mock_configure, mock_model_class):
        """Test chat with custom temperature."""
        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.usage_metadata = Mock(prompt_character_count=5, candidates_token_count=10)
        mock_response.candidates = [Mock(finish_reason=Mock(name="STOP"))]

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="test-key")
        provider = GoogleProvider(config)

        provider.chat("Message", temperature=0.2)

        call_kwargs = mock_model.generate_content.call_args[1]
        assert call_kwargs["generation_config"]["temperature"] == 0.2

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_chat_invokes_usage_callback(self, mock_configure, mock_model_class):
        """Test chat invokes usage callbacks."""
        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.usage_metadata = Mock(prompt_character_count=100, candidates_token_count=50)
        mock_response.candidates = [Mock(finish_reason=Mock(name="STOP"))]

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="test-key")
        provider = GoogleProvider(config)

        callback = Mock()
        provider.add_usage_callback(callback)

        provider.chat("Message")

        callback.assert_called_once()
        usage_arg = callback.call_args[0][0]
        assert isinstance(usage_arg, TokenUsage)
        assert usage_arg.input_tokens == 100


class TestGoogleAsync:
    """Test Google async methods."""

    @pytest.mark.asyncio
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    async def test_achat_basic_message(self, mock_configure, mock_model_class):
        """Test basic async chat message."""
        mock_response = Mock()
        mock_response.text = "Hello async!"
        mock_response.usage_metadata = Mock(prompt_character_count=10, candidates_token_count=20)
        mock_response.candidates = [Mock(finish_reason=Mock(name="STOP"))]

        mock_model = AsyncMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_model_class.return_value = mock_model

        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="test-key")
        provider = GoogleProvider(config)

        response = await provider.achat("Hello!")

        assert response.content == "Hello async!"


class TestGoogleErrorHandling:
    """Test Google error handling."""

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_handle_google_error_rate_limit(self, mock_configure, mock_model_class):
        """Test rate limit error handling."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("429 rate_limit_exceeded")
        mock_model_class.return_value = mock_model

        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="test-key")
        provider = GoogleProvider(config)

        with pytest.raises(RateLimitError):
            provider.chat("Message")

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_handle_google_error_invalid_api_key(self, mock_configure, mock_model_class):
        """Test invalid API key error handling."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("invalid_api_key")
        mock_model_class.return_value = mock_model

        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="invalid")
        provider = GoogleProvider(config)

        with pytest.raises(AuthenticationError):
            provider.chat("Message")


class TestGoogleRepresentation:
    """Test Google provider representation."""

    def test_repr(self):
        """Test provider representation."""
        config = LLMConfig(provider="google", model="gemini-1.5-pro", api_key="test-key")
        provider = GoogleProvider(config)
        repr_str = repr(provider)

        assert "GoogleProvider" in repr_str
        assert "gemini-1.5-pro" in repr_str
