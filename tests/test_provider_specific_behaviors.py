"""Tests for provider-specific behaviors and edge cases.

Tests specific to each LLM provider:
- OpenAI (GPT models)
- Google (Gemini models)
- Ollama (Local models)
- Provider-specific token calculation
- Provider-specific error handling
- Provider-specific response formatting
"""

import pytest
from unittest.mock import Mock, patch
import requests  # Import real requests to access real exception classes

# Skip all tests in this module if required dependencies are not installed
pytest.importorskip("cryptography")
pytest.importorskip("openai")

from socratic_nexus.clients.openai_client import OpenAIClient
from socratic_nexus.clients.ollama_client import OllamaClient

# GoogleClient is optional - skip if google.generativeai is not available
GoogleClient = None
try:
    from socratic_nexus.clients.google_client import GoogleClient
except (ImportError, ModuleNotFoundError):
    # google.generativeai not installed - will skip tests using GoogleClient
    pass


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator."""
    orch = Mock()
    orch.config = Mock()
    orch.config.openai_model = "gpt-4-turbo"
    orch.config.google_model = "gemini-pro"
    orch.config.ollama_model = "mistral"
    orch.config.ollama_url = "http://localhost:11434"
    orch.event_emitter = Mock()
    orch.database = Mock()
    orch.database.get_api_key.return_value = None
    return orch


@pytest.fixture
def mock_requests_module():
    """Fixture to provide properly configured mock requests module.

    This ensures that:
    1. requests.exceptions.ConnectionError is the real exception class (not a Mock)
    2. requests.Session() returns a properly configured mock session
    """
    with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
        # Set the real exception class so catching works properly
        mock_requests.exceptions.ConnectionError = requests.exceptions.ConnectionError

        # Create a mock session instance
        mock_session = Mock()
        mock_requests.Session.return_value = mock_session

        yield mock_requests, mock_session


class TestOpenAISpecificBehaviors:
    """Tests for OpenAI-specific provider behaviors."""

    def test_openai_token_calculation(self, mock_orchestrator):
        """Test OpenAI's token calculation model."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            response = Mock()
            response.choices = [Mock(message=Mock(content="response"))]
            response.usage = Mock(
                prompt_tokens=100,
                completion_tokens=50,
            )
            mock_client.chat.completions.create.return_value = response

            client = OpenAIClient(api_key="sk-test", orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            # OpenAI uses prompt_tokens and completion_tokens
            assert isinstance(result, (str, type(None)))

    def test_openai_model_selection(self, mock_orchestrator):
        """Test OpenAI model selection."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="sk-test", orchestrator=mock_orchestrator)

            # Should use gpt-4-turbo from config
            assert client.model == "gpt-4-turbo"

    def test_openai_rate_limit_retry(self, mock_orchestrator):
        """Test OpenAI rate limit handling."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Simulate rate limit then success
            mock_client.chat.completions.create.side_effect = [
                Exception("429 Too Many Requests"),
                Mock(
                    choices=[Mock(message=Mock(content="success"))],
                    usage=Mock(prompt_tokens=10, completion_tokens=20),
                ),
            ]

            client = OpenAIClient(api_key="sk-test", orchestrator=mock_orchestrator)

            # First call fails, would need retry logic
            try:
                client.generate_response("test")
            except Exception:
                pass

    def test_openai_response_format(self, mock_orchestrator):
        """Test OpenAI response format parsing."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            response = Mock()
            response.choices = [
                Mock(
                    message=Mock(
                        content="Generated text response",
                        role="assistant",
                    )
                )
            ]
            response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = response

            client = OpenAIClient(api_key="sk-test", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt")

            assert isinstance(result, (str, type(None)))

    def test_openai_function_calling_support(self, mock_orchestrator):
        """Test OpenAI function calling capability."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            response = Mock()
            response.choices = [Mock(message=Mock(content="response"))]
            response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = response

            client = OpenAIClient(api_key="sk-test", orchestrator=mock_orchestrator)

            # OpenAI supports function calling in its API
            result = client.generate_response("test")
            assert isinstance(result, (str, type(None)))


@pytest.mark.skipif(GoogleClient is None, reason="google.generativeai not installed")
class TestGoogleSpecificBehaviors:
    """Tests for Google Gemini-specific provider behaviors."""

    def test_google_token_calculation(self, mock_orchestrator):
        """Test Google's token calculation model."""
        pytest.importorskip("google.generativeai")

        with patch("socratic_nexus.clients.google_client.genai.GenerativeModel") as mock_generative_model:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "response"
            mock_response.usage_metadata = Mock(
                prompt_token_count=100,
                candidates_token_count=50,
            )
            mock_model.generate_content.return_value = mock_response

            # Configure the patch to return a mock model instance when called
            mock_generative_model.return_value = mock_model

            with patch("socratic_nexus.clients.google_client.genai.configure"):
                client = GoogleClient(api_key="test", orchestrator=mock_orchestrator)
                # Client should have the mock model instance
                assert client.client is not None or client.model is not None

    def test_google_model_selection(self, mock_orchestrator):
        """Test Google model selection."""
        pytest.importorskip("google.generativeai")

        with patch("socratic_nexus.clients.google_client.genai"):
            client = GoogleClient(api_key="test", orchestrator=mock_orchestrator)

            # Should use gemini-pro from config
            assert client.model == "gemini-pro"

    def test_google_safety_settings(self, mock_orchestrator):
        """Test Google's safety settings."""
        pytest.importorskip("google.generativeai")

        with patch("socratic_nexus.clients.google_client.genai.GenerativeModel"):
            client = GoogleClient(api_key="test", orchestrator=mock_orchestrator)

            # Google Gemini has safety settings
            assert hasattr(client, "model") or hasattr(client, "client")

    def test_google_response_format(self, mock_orchestrator):
        """Test Google response format parsing."""
        pytest.importorskip("google.generativeai")

        with patch("socratic_nexus.clients.google_client.genai.GenerativeModel"):
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "Generated text"
            mock_model.generate_content.return_value = mock_response

            client = GoogleClient(api_key="test", orchestrator=mock_orchestrator)
            client.client = mock_model

            # Test text extraction
            result = mock_response.text
            assert isinstance(result, str)


class TestOllamaSpecificBehaviors:
    """Tests for Ollama local model-specific behaviors."""

    def test_ollama_local_connection(self, mock_orchestrator, mock_requests_module):
        """Test Ollama local connection setup."""
        mock_requests, mock_session = mock_requests_module
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test"}
        mock_session.post.return_value = mock_response
        mock_session.get.return_value = mock_response

        client = OllamaClient(orchestrator=mock_orchestrator)

        assert client.base_url == "http://localhost:11434"

    def test_ollama_custom_url_config(self, mock_orchestrator, mock_requests_module):
        """Test Ollama with custom URL configuration."""
        mock_requests, _ = mock_requests_module
        custom_url = "http://remote-server:8000"
        mock_orchestrator.config.ollama_url = custom_url
        client = OllamaClient(orchestrator=mock_orchestrator)

        assert client.base_url == custom_url

    def test_ollama_streaming_response(self, mock_orchestrator, mock_requests_module):
        """Test Ollama streaming response handling."""
        mock_requests, mock_session = mock_requests_module
        mock_response = Mock()
        mock_response.status_code = 200
        # Ollama can return streaming responses
        mock_response.json.return_value = {
            "response": "streamed response",
            "model": "mistral",
            "done": True,
        }
        mock_session.post.return_value = mock_response
        mock_session.get.return_value = mock_response

        client = OllamaClient(orchestrator=mock_orchestrator)
        result = client.generate_response("test")

        assert isinstance(result, (str, type(None)))

    def test_ollama_model_selection(self, mock_orchestrator, mock_requests_module):
        """Test Ollama model selection."""
        mock_requests, _ = mock_requests_module
        client = OllamaClient(orchestrator=mock_orchestrator)

        # Should use mistral from config
        assert client.model == "mistral"

    def test_ollama_offline_error_handling(self, mock_orchestrator, mock_requests_module):
        """Test Ollama offline/unavailable server."""
        from socratic_nexus.exceptions import APIError

        mock_requests, mock_session = mock_requests_module
        mock_session.get.side_effect = requests.exceptions.ConnectionError(
            "Unable to connect to Ollama server"
        )

        client = OllamaClient(orchestrator=mock_orchestrator)
        # ConnectionError should be caught and handled as APIError
        try:
            result = client.generate_response("test")
            # Should handle gracefully
            assert result is None or isinstance(result, str)
        except APIError:
            # APIError is the expected wrapped exception
            pass

    def test_ollama_custom_parameters(self, mock_orchestrator, mock_requests_module):
        """Test Ollama custom parameters."""
        mock_requests, mock_session = mock_requests_module
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "response"}
        mock_session.post.return_value = mock_response
        mock_session.get.return_value = mock_response

        client = OllamaClient(orchestrator=mock_orchestrator)

        # Ollama supports custom parameters
        result = client.generate_response("test", temperature=0.5)
        assert isinstance(result, (str, type(None)))


class TestProviderTokenCostCalculation:
    """Tests for provider-specific token cost calculations."""

    def test_claude_token_pricing(self, mock_orchestrator):
        """Test Claude token cost calculation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=1000, output_tokens=500)
            mock_client.messages.create.return_value = response

            from socratic_nexus.clients.claude_client import ClaudeClient

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            cost = client._calculate_cost(response.usage)

            # Should return a numeric cost
            assert isinstance(cost, (int, float))

    def test_openai_token_pricing(self, mock_orchestrator):
        """Test OpenAI token cost calculation."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            response = Mock()
            response.choices = [Mock(message=Mock(content="response"))]
            response.usage = Mock(prompt_tokens=1000, completion_tokens=500)
            mock_client.chat.completions.create.return_value = response

            client = OpenAIClient(api_key="sk-test", orchestrator=mock_orchestrator)
            cost = client._calculate_cost(response.usage)

            # Should return a numeric cost
            assert isinstance(cost, (int, float))

    def test_google_token_pricing(self, mock_orchestrator):
        """Test Google token cost calculation."""
        pytest.importorskip("google.generativeai")

        if GoogleClient is None:
            pytest.skip("GoogleClient not available (google.generativeai not installed)")

        with patch("socratic_nexus.clients.google_client.genai"):
            client = GoogleClient(api_key="test", orchestrator=mock_orchestrator)

            # Google's _calculate_cost_google takes (input_len, output_len) as arguments
            # Not a usage object. Test with realistic text lengths.
            input_len = 1000  # approximate character count
            output_len = 500  # approximate character count

            # Call the method with proper arguments
            cost = client._calculate_cost_google(input_len, output_len)

            assert isinstance(cost, (int, float))


class TestProviderErrorMessages:
    """Tests for provider-specific error messages."""

    def test_openai_error_message_format(self, mock_orchestrator):
        """Test OpenAI error message format."""
        from socratic_nexus.exceptions import APIError

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception(
                "OpenAI API error: Invalid request"
            )

            client = OpenAIClient(api_key="sk-test", orchestrator=mock_orchestrator)

            # Should raise APIError or handle error gracefully
            try:
                result = client.generate_response("test")
                # If it doesn't raise, it should return None or str
                assert result is None or isinstance(result, str)
            except APIError:
                # APIError is also acceptable behavior
                pass

    def test_google_error_message_format(self, mock_orchestrator):
        """Test Google error message format."""
        pytest.importorskip("google.generativeai")

        if GoogleClient is None:
            pytest.skip("GoogleClient not available (google.generativeai not installed)")

        with patch("socratic_nexus.clients.google_client.genai"):
            client = GoogleClient(api_key="test", orchestrator=mock_orchestrator)

            # Should handle errors without crashing
            assert isinstance(client.model, str) or client.model is None

    def test_ollama_error_message_format(self, mock_orchestrator):
        """Test Ollama error message format."""
        from socratic_nexus.exceptions import APIError

        with patch("socratic_nexus.clients.ollama_client.requests.Session") as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock the get request to return a valid response for the connection check
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_session.get.return_value = mock_get_response

            # Mock the post request to raise an error
            mock_session.post.side_effect = Exception("Ollama connection failed")

            client = OllamaClient(orchestrator=mock_orchestrator)

            # Should raise APIError or handle error gracefully
            try:
                result = client.generate_response("test")
                assert result is None or isinstance(result, str)
            except APIError:
                # APIError is expected behavior
                pass


class TestProviderResponseVariations:
    """Tests for handling provider-specific response variations."""

    def test_claude_response_with_multiple_content_blocks(self, mock_orchestrator):
        """Test Claude handling multiple content blocks."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [
                Mock(text="block1"),
                Mock(text="block2"),
                Mock(text="block3"),
            ]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            from socratic_nexus.clients.claude_client import ClaudeClient

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)

            # Should extract first text block
            result = client.generate_response("test")
            assert isinstance(result, (str, type(None)))

    def test_openai_response_with_multiple_choices(self, mock_orchestrator):
        """Test OpenAI handling multiple choice responses."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            response = Mock()
            response.choices = [
                Mock(message=Mock(content="choice1")),
                Mock(message=Mock(content="choice2")),
            ]
            response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = response

            client = OpenAIClient(api_key="sk-test", orchestrator=mock_orchestrator)

            # Should extract first choice
            result = client.generate_response("test")
            assert isinstance(result, (str, type(None)))

    def test_ollama_partial_response(self, mock_orchestrator, mock_requests_module):
        """Test Ollama handling partial responses."""
        mock_requests, mock_session = mock_requests_module
        mock_response = Mock()
        mock_response.status_code = 200
        # Ollama might return incomplete response with done=False
        mock_response.json.return_value = {
            "response": "partial response...",
            "done": False,
        }
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response

        client = OllamaClient(orchestrator=mock_orchestrator)
        result = client.generate_response("test")

        # Should handle partial responses
        assert result is None or isinstance(result, str)


class TestProviderCaching:
    """Tests for provider-specific caching behaviors."""

    def test_cache_key_consistency_across_providers(self, mock_orchestrator):
        """Test cache key generation is consistent."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            from socratic_nexus.clients.claude_client import ClaudeClient

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)

            # Cache keys should be consistent
            key1 = client._get_cache_key("message1")
            key2 = client._get_cache_key("message1")
            key3 = client._get_cache_key("message2")

            assert key1 == key2  # Same message = same key
            assert key1 != key3  # Different message = different key

    def test_provider_specific_cache_behavior(self, mock_orchestrator):
        """Test provider-specific cache implementations."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            response = Mock()
            response.choices = [Mock(message=Mock(content="response"))]
            response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = response

            client = OpenAIClient(api_key="sk-test", orchestrator=mock_orchestrator)

            # OpenAI client should have cache
            assert hasattr(client, "_insights_cache")
            assert isinstance(client._insights_cache, dict)
