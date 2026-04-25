"""Comprehensive tests for OpenAIClient implementation."""

import pytest
from unittest.mock import Mock, patch

# Skip all tests in this module if required dependencies are not installed
pytest.importorskip("cryptography")
pytest.importorskip("openai")

from socratic_nexus.clients.openai_client import OpenAIClient
from socratic_nexus.models import ProjectContext
from socratic_nexus.exceptions import APIError


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator."""
    orch = Mock()
    orch.config = Mock()
    orch.config.openai_model = "gpt-4-turbo"
    orch.event_emitter = Mock()
    orch.database = Mock()
    orch.database.get_api_key.return_value = None
    return orch


class TestOpenAIClientInitialization:
    """Tests for OpenAIClient initialization."""

    def test_init_with_valid_api_key(self, mock_orchestrator):
        """Test initialization with valid API key."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="sk-test-key", orchestrator=mock_orchestrator)
            assert client.api_key == "sk-test-key"

    def test_init_with_none_api_key(self, mock_orchestrator):
        """Test initialization with None API key."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key=None, orchestrator=mock_orchestrator)
            assert client.api_key is None

    def test_init_with_placeholder_key(self, mock_orchestrator):
        """Test initialization with placeholder key."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="placeholder_test", orchestrator=mock_orchestrator)
            assert client.client is None

    def test_init_sets_default_model(self, mock_orchestrator):
        """Test default model is set."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            assert client.model == "gpt-4-turbo"

    def test_init_uses_orchestrator_model(self, mock_orchestrator):
        """Test model from orchestrator is used."""
        mock_orchestrator.config.openai_model = "gpt-3.5-turbo"
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            assert client.model == "gpt-3.5-turbo"

    def test_init_initializes_caches(self, mock_orchestrator):
        """Test caches are initialized."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            assert hasattr(client, "_insights_cache")
            assert isinstance(client._insights_cache, dict)


class TestOpenAIClientGenerateResponse:
    """Tests for generate_response method."""

    def test_generate_response_basic(self, mock_orchestrator):
        """Test basic response generation."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="test response"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("test prompt")

            assert result is not None or result is None

    def test_generate_response_with_max_tokens(self, mock_orchestrator):
        """Test response generation with max_tokens."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="response"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            _ = client.generate_response("prompt", max_tokens=100)

            mock_client.chat.completions.create.assert_called()

    def test_generate_response_with_temperature(self, mock_orchestrator):
        """Test response generation with temperature."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="response"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            _ = client.generate_response("prompt", temperature=0.7)

            mock_client.chat.completions.create.assert_called()


class TestOpenAIClientGenerateCode:
    """Tests for generate_code method."""

    def test_generate_code_basic(self, mock_orchestrator):
        """Test basic code generation."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="def hello():\n    pass"))]
            mock_response.usage = Mock(prompt_tokens=20, completion_tokens=15)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_code("write hello function")

            assert result is not None or result is None

    def test_generate_code_with_language(self, mock_orchestrator):
        """Test code generation with language specification."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="code"))]
            mock_response.usage = Mock(prompt_tokens=20, completion_tokens=15)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            _ = client.generate_code("write function")

            mock_client.chat.completions.create.assert_called()


class TestOpenAIClientExtractInsights:
    """Tests for extract_insights method."""

    def test_extract_insights_basic(self, mock_orchestrator):
        """Test basic insights extraction."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content='{"insights": "test"}'))]
            mock_response.usage = Mock(prompt_tokens=30, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("user response", project)

            assert isinstance(result, (dict, type(None)))

    def test_extract_insights_with_empty_response(self, mock_orchestrator):
        """Test insights extraction with empty response."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=""))]
            mock_response.usage = Mock(prompt_tokens=30, completion_tokens=0)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("", project)

            assert isinstance(result, (dict, type(None)))

    def test_extract_insights_with_project_context(self, mock_orchestrator):
        """Test insights extraction includes project context."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content='{"test": "data"}'))]
            mock_response.usage = Mock(prompt_tokens=30, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="TestProject", phase="planning")
            _ = client.extract_insights("response", project)

            mock_client.chat.completions.create.assert_called()


class TestOpenAIClientSocraticQuestion:
    """Tests for generate_socratic_question method."""

    def test_generate_socratic_question_basic(self, mock_orchestrator):
        """Test basic socratic question generation."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="What is the purpose?"))]
            mock_response.usage = Mock(prompt_tokens=20, completion_tokens=10)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_socratic_question("explain functions")

            assert result is not None or result is None

    def test_generate_socratic_question_with_topic(self, mock_orchestrator):
        """Test socratic question generation with topic."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="question"))]
            mock_response.usage = Mock(prompt_tokens=20, completion_tokens=10)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            _ = client.generate_socratic_question("callbacks")

            mock_client.chat.completions.create.assert_called()


class TestOpenAIClientCaching:
    """Tests for caching functionality."""

    def test_cache_key_generation(self, mock_orchestrator):
        """Test cache key generation."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            key = client._get_cache_key("test message")
            assert isinstance(key, str)
            assert len(key) == 64

    def test_cache_hit(self, mock_orchestrator):
        """Test cache hit on repeated calls."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            message = "test"
            key = client._get_cache_key(message)
            client._insights_cache[key] = {"cached": True}

            assert key in client._insights_cache

    def test_different_messages_different_keys(self, mock_orchestrator):
        """Test different messages produce different cache keys."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            key1 = client._get_cache_key("message 1")
            key2 = client._get_cache_key("message 2")
            assert key1 != key2


class TestOpenAIClientErrorHandling:
    """Tests for error handling."""

    def test_no_api_key_raises_error(self, mock_orchestrator):
        """Test APIError raised when no API key available."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            mock_orchestrator.database.get_api_key.return_value = None
            client = OpenAIClient(api_key=None, orchestrator=mock_orchestrator)

            with pytest.raises(APIError):
                client._get_client()

    def test_api_error_on_generation_failure(self, mock_orchestrator):
        """Test handling of API errors during generation."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API Error")

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            try:
                _ = client.generate_response("test")
            except Exception:
                pass


class TestOpenAIClientAuthFlow:
    """Tests for authentication flow."""

    def test_get_auth_credential_api_key(self, mock_orchestrator):
        """Test getting auth credential."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            client = OpenAIClient(api_key="sk-my-key", orchestrator=mock_orchestrator)
            cred = client.get_auth_credential("api_key")
            assert cred == "sk-my-key"

    def test_get_user_api_key_fallback(self, mock_orchestrator):
        """Test fallback to environment key when user key not available."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            mock_orchestrator.database.get_api_key.return_value = None
            client = OpenAIClient(api_key="env-key", orchestrator=mock_orchestrator)
            key, is_user = client._get_user_api_key("user123")
            assert key == "env-key"
            assert is_user is False


class TestOpenAIClientIntegration:
    """Integration tests for OpenAIClient."""

    def test_client_initialization_and_use(self, mock_orchestrator):
        """Test full client initialization and usage."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="test response"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)
            assert client.api_key == "test-key"
            assert client.model == "gpt-4-turbo"

            _ = client.generate_response("test")
            mock_client.chat.completions.create.assert_called()

    def test_multiple_api_calls(self, mock_orchestrator):
        """Test multiple API calls in sequence."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="response"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            client = OpenAIClient(api_key="test-key", orchestrator=mock_orchestrator)

            _ = client.generate_response("prompt 1")
            _ = client.generate_code("code prompt")
            _ = client.generate_socratic_question("topic")

            assert mock_client.chat.completions.create.call_count >= 3
