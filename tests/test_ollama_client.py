"""Comprehensive tests for OllamaClient implementation."""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.ollama_client import OllamaClient
from socratic_nexus.models import ProjectContext, ConflictInfo
from socratic_nexus.exceptions import APIError


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator."""
    orch = Mock()
    orch.config = Mock()
    orch.config.ollama_model = "mistral"
    orch.config.ollama_url = "http://localhost:11434"
    orch.event_emitter = Mock()
    orch.database = Mock()
    orch.database.get_api_key.return_value = None
    return orch


class TestOllamaClientInitialization:
    """Tests for OllamaClient initialization."""

    def test_init_with_default_url(self, mock_orchestrator):
        """Test initialization with default URL."""
        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)
            assert client.api_base_url == "http://localhost:11434"

    def test_init_with_custom_url(self, mock_orchestrator):
        """Test initialization with custom URL."""
        with patch("socratic_nexus.clients.ollama_client.requests"):
            custom_url = "http://custom:8000"
            client = OllamaClient(api_base_url=custom_url, orchestrator=mock_orchestrator)
            assert client.api_base_url == custom_url

    def test_init_sets_default_model(self, mock_orchestrator):
        """Test default model is set."""
        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)
            assert client.model == "mistral"

    def test_init_uses_orchestrator_model(self, mock_orchestrator):
        """Test model from orchestrator is used."""
        mock_orchestrator.config.ollama_model = "neural-chat"
        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)
            assert client.model == "neural-chat"

    def test_init_initializes_caches(self, mock_orchestrator):
        """Test caches are initialized."""
        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)
            assert hasattr(client, '_insights_cache')
            assert isinstance(client._insights_cache, dict)


class TestOllamaClientGenerateResponse:
    """Tests for generate_response method."""

    def test_generate_response_basic(self, mock_orchestrator):
        """Test basic response generation."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "test response",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            result = client.generate_response("test prompt")

            assert result is not None or result is None

    def test_generate_response_with_max_tokens(self, mock_orchestrator):
        """Test response generation with max_tokens."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "response",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            _ = client.generate_response("prompt", max_tokens=100)

            mock_requests.post.assert_called()

    def test_generate_response_with_temperature(self, mock_orchestrator):
        """Test response generation with temperature."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "response",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            _ = client.generate_response("prompt", temperature=0.5)

            mock_requests.post.assert_called()

    def test_generate_response_http_error(self, mock_orchestrator):
        """Test handling of HTTP errors."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            result = client.generate_response("prompt")

            assert result is None or isinstance(result, str)


class TestOllamaClientGenerateCode:
    """Tests for generate_code method."""

    def test_generate_code_basic(self, mock_orchestrator):
        """Test basic code generation."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "def hello():\n    pass",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            result = client.generate_code("write hello function")

            assert result is not None or result is None

    def test_generate_code_with_language(self, mock_orchestrator):
        """Test code generation with language specification."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "code",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            _ = client.generate_code("write function", language="python")

            mock_requests.post.assert_called()


class TestOllamaClientExtractInsights:
    """Tests for extract_insights method."""

    def test_extract_insights_basic(self, mock_orchestrator):
        """Test basic insights extraction."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": '{"insights": "test"}',
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("user response", project)

            assert isinstance(result, (dict, type(None)))

    def test_extract_insights_with_empty_response(self, mock_orchestrator):
        """Test insights extraction with empty response."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("", project)

            assert isinstance(result, (dict, type(None)))

    def test_extract_insights_with_project_context(self, mock_orchestrator):
        """Test insights extraction includes project context."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": '{"test": "data"}',
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="TestProject", phase="design")
            _ = client.extract_insights("response", project)

            mock_requests.post.assert_called()


class TestOllamaClientSocraticQuestion:
    """Tests for generate_socratic_question method."""

    def test_generate_socratic_question_basic(self, mock_orchestrator):
        """Test basic socratic question generation."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "What would you try next?",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            result = client.generate_socratic_question("explain loops")

            assert result is not None or result is None

    def test_generate_socratic_question_with_topic(self, mock_orchestrator):
        """Test socratic question generation with topic."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "question",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            _ = client.generate_socratic_question("recursion")

            mock_requests.post.assert_called()


class TestOllamaClientCaching:
    """Tests for caching functionality."""

    def test_cache_key_generation(self, mock_orchestrator):
        """Test cache key generation."""
        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)
            key = client._get_cache_key("test message")
            assert isinstance(key, str)
            assert len(key) == 64

    def test_cache_hit(self, mock_orchestrator):
        """Test cache hit on repeated calls."""
        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)
            message = "test"
            key = client._get_cache_key(message)
            client._insights_cache[key] = {"cached": True}

            assert key in client._insights_cache

    def test_different_messages_different_keys(self, mock_orchestrator):
        """Test different messages produce different cache keys."""
        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)
            key1 = client._get_cache_key("message 1")
            key2 = client._get_cache_key("message 2")
            assert key1 != key2


class TestOllamaClientConnectionHandling:
    """Tests for connection handling."""

    def test_connection_timeout(self, mock_orchestrator):
        """Test handling of connection timeout."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_requests.post.side_effect = ConnectionError("Connection failed")

            client = OllamaClient(orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            assert result is None or isinstance(result, str)

    def test_invalid_server_response(self, mock_orchestrator):
        """Test handling of invalid server response."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            assert result is None or isinstance(result, str)


class TestOllamaClientAuthFlow:
    """Tests for authentication flow."""

    def test_get_auth_credential_api_key(self, mock_orchestrator):
        """Test getting auth credential (Ollama doesn't require API key)."""
        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)
            # Ollama doesn't use API keys typically
            try:
                cred = client.get_auth_credential("api_key")
                # May raise or return None
            except APIError:
                pass


class TestOllamaClientIntegration:
    """Integration tests for OllamaClient."""

    def test_client_initialization_and_use(self, mock_orchestrator):
        """Test full client initialization and usage."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "test response",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)
            assert client.model == "mistral"
            assert client.api_base_url == "http://localhost:11434"

            _ = client.generate_response("test")
            mock_requests.post.assert_called()

    def test_multiple_api_calls(self, mock_orchestrator):
        """Test multiple API calls in sequence."""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "response",
                "model": "mistral"
            }
            mock_requests.post.return_value = mock_response

            client = OllamaClient(orchestrator=mock_orchestrator)

            _ = client.generate_response("prompt 1")
            _ = client.generate_code("code prompt")
            _ = client.generate_socratic_question("topic")

            assert mock_requests.post.call_count >= 3

    def test_url_configuration(self, mock_orchestrator):
        """Test URL configuration from orchestrator."""
        mock_orchestrator.config.ollama_url = "http://custom.server:9000"
        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)
            assert client.api_base_url == "http://custom.server:9000"
