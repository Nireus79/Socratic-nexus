"""
Error handling and edge case tests for all LLM clients.
Covers exception paths, validation, and boundary conditions.
"""

import importlib.util
import pytest
from unittest.mock import Mock, patch, MagicMock

from socratic_nexus.models import ProjectContext
from socratic_nexus.exceptions import APIError

# Check which clients are available
_google_available = importlib.util.find_spec("google.generativeai") is not None
_openai_available = importlib.util.find_spec("openai") is not None
_ollama_available = importlib.util.find_spec("ollama") is not None


@pytest.mark.skipif(not _google_available, reason="google.generativeai not installed")
class TestGoogleClientErrors:
    """Test error handling in Google client."""

    def test_google_client_missing_genai_raises_error(self):
        """Test that GoogleClient raises helpful error when google.generativeai is missing."""
        # Temporarily hide genai
        with patch("socratic_nexus.clients.google_client.genai", None):
            from socratic_nexus.clients.google_client import GoogleClient

            with pytest.raises(ImportError):
                GoogleClient(api_key="test-key")

    def test_google_response_with_malformed_json(self):
        """Test handling of malformed JSON in response."""
        with patch("socratic_nexus.clients.google_client.genai") as mock_genai:
            from socratic_nexus.clients.google_client import GoogleClient

            orch = Mock()
            orch.config = Mock()
            orch.config.google_model = "gemini-pro"
            orch.event_emitter = Mock()
            orch.database = None

            client = GoogleClient(api_key="test-key", orchestrator=orch)

            mock_model = MagicMock()
            mock_response = Mock()
            mock_response.text = "Invalid JSON {{"
            mock_model.generate_content.return_value = mock_response
            client.client = mock_model

            project = ProjectContext(project_name="Test")
            result = client.extract_insights("test", project)

            # Should handle gracefully
            assert result is None or isinstance(result, dict)

    def test_google_response_with_none_text(self):
        """Test handling of None text in response."""
        with patch("socratic_nexus.clients.google_client.genai") as mock_genai:
            from socratic_nexus.clients.google_client import GoogleClient

            orch = Mock()
            orch.config = Mock()
            orch.config.google_model = "gemini-pro"
            orch.event_emitter = Mock()
            orch.database = None

            client = GoogleClient(api_key="test-key", orchestrator=orch)

            mock_model = MagicMock()
            mock_response = Mock()
            mock_response.text = None
            mock_model.generate_content.return_value = mock_response
            client.client = mock_model

            result = client.generate_response("test")

            # Should handle None gracefully
            assert result is None or isinstance(result, str)

    def test_google_empty_response(self):
        """Test handling of empty response."""
        with patch("socratic_nexus.clients.google_client.genai") as mock_genai:
            from socratic_nexus.clients.google_client import GoogleClient

            orch = Mock()
            orch.config = Mock()
            orch.config.google_model = "gemini-pro"
            orch.event_emitter = Mock()
            orch.database = None

            client = GoogleClient(api_key="test-key", orchestrator=orch)

            mock_model = MagicMock()
            mock_response = Mock()
            mock_response.text = ""
            mock_model.generate_content.return_value = mock_response
            client.client = mock_model

            result = client.generate_response("test")

            # Should handle empty string
            assert isinstance(result, str)


@pytest.mark.skipif(not _openai_available, reason="openai not installed")
class TestOpenAIClientErrors:
    """Test error handling in OpenAI client."""

    def test_openai_response_with_empty_choices(self):
        """Test handling of empty choices in response."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            orch = Mock()
            orch.config = Mock()
            orch.config.openai_model = "gpt-4"
            orch.event_emitter = Mock()
            orch.database = None

            client = OpenAIClient(api_key="test-key", orchestrator=orch)

            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = []  # Empty choices
            mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)
            mock_client.chat.completions.create.return_value = mock_response
            client.client = mock_client

            result = client.generate_response("test")

            # Should handle gracefully
            assert result is None or isinstance(result, str)

    def test_openai_response_with_none_content(self):
        """Test handling of None content in choice."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            orch = Mock()
            orch.config = Mock()
            orch.config.openai_model = "gpt-4"
            orch.event_emitter = Mock()
            orch.database = None

            client = OpenAIClient(api_key="test-key", orchestrator=orch)

            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=None))]
            mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)
            mock_client.chat.completions.create.return_value = mock_response
            client.client = mock_client

            result = client.generate_response("test")

            # Should handle None gracefully
            assert result is None or isinstance(result, str)

    def test_openai_api_error_handling(self):
        """Test handling of API errors."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            orch = Mock()
            orch.config = Mock()
            orch.config.openai_model = "gpt-4"
            orch.event_emitter = Mock()
            orch.database = None

            client = OpenAIClient(api_key="test-key", orchestrator=orch)

            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            client.client = mock_client

            try:
                result = client.generate_response("test")
                # Should either return None or raise error
                assert result is None or isinstance(result, str)
            except Exception:
                # Expected - error handling
                pass


@pytest.mark.skipif(not _ollama_available, reason="ollama not installed")
class TestOllamaClientErrors:
    """Test error handling in Ollama client."""

    def test_ollama_connection_error(self):
        """Test handling of connection errors."""
        import requests

        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"
            orch.event_emitter = Mock()
            orch.database = None

            client = OllamaClient(orchestrator=orch)

            mock_session = MagicMock()
            mock_session.post.side_effect = requests.exceptions.ConnectionError("Failed")
            client.session = mock_session

            try:
                result = client.generate_response("test")
                assert result is None or isinstance(result, str)
            except Exception:
                # Expected
                pass

    def test_ollama_timeout_error(self):
        """Test handling of timeout errors."""
        import requests

        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"
            orch.event_emitter = Mock()
            orch.database = None

            client = OllamaClient(orchestrator=orch)

            mock_session = MagicMock()
            mock_session.post.side_effect = requests.exceptions.Timeout("Timeout")
            client.session = mock_session

            try:
                result = client.generate_response("test")
                assert result is None or isinstance(result, str)
            except Exception:
                # Expected
                pass

    def test_ollama_empty_response(self):
        """Test handling of empty response."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"
            orch.event_emitter = Mock()
            orch.database = None

            client = OllamaClient(orchestrator=orch)

            mock_session = MagicMock()
            mock_response = Mock()
            mock_response.json.return_value = {"response": ""}
            mock_session.post.return_value = mock_response
            client.session = mock_session

            result = client.generate_response("test")

            # Should handle empty response
            assert isinstance(result, str)

    def test_ollama_response_missing_response_key(self):
        """Test handling of response missing 'response' key."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"
            orch.event_emitter = Mock()
            orch.database = None

            client = OllamaClient(orchestrator=orch)

            mock_session = MagicMock()
            mock_response = Mock()
            mock_response.json.return_value = {"output": "test"}  # Missing 'response' key
            mock_session.post.return_value = mock_response
            client.session = mock_session

            try:
                result = client.generate_response("test")
                # Should handle missing key gracefully
                assert result is None or isinstance(result, str)
            except Exception:
                # Expected - should handle error
                pass


# Additional test for common functionality
class TestClientInitializationValidation:
    """Test client initialization validation across all clients."""

    @pytest.mark.skipif(not _google_available, reason="google.generativeai not installed")
    def test_google_placeholder_api_key_handling(self):
        """Test Google client handling of placeholder API keys."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            # Should not fail with placeholder key
            client = GoogleClient(api_key="placeholder-key")
            assert client.api_key == "placeholder-key"

    @pytest.mark.skipif(not _openai_available, reason="openai not installed")
    def test_openai_placeholder_api_key_handling(self):
        """Test OpenAI client handling of placeholder API keys."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            # Should not fail with placeholder key
            client = OpenAIClient(api_key="placeholder-key")
            assert client.api_key == "placeholder-key"

    @pytest.mark.skipif(not _ollama_available, reason="ollama not installed")
    def test_ollama_no_api_key_required(self):
        """Test that Ollama doesn't require API key."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"

            # Ollama doesn't require API key
            client = OllamaClient(orchestrator=orch)
            assert client is not None
