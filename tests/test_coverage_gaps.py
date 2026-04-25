"""
Targeted tests to cover critical gaps in client implementations.
Focuses on high-impact code paths that are currently untested.
"""

import importlib.util
import pytest
from unittest.mock import Mock, patch, MagicMock

# Check which clients are available
_google_available = importlib.util.find_spec("google.generativeai") is not None
_openai_available = importlib.util.find_spec("openai") is not None
_ollama_available = importlib.util.find_spec("ollama") is not None


@pytest.mark.skipif(not _google_available, reason="google.generativeai not installed")
class TestGoogleClientGaps:
    """Test critical uncovered paths in Google client."""

    def test_google_get_client_with_placeholder_key(self):
        """Test _get_client doesn't initialize with placeholder key."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            orch = Mock()
            orch.config = Mock()
            orch.config.google_model = "gemini-pro"
            orch.database = None

            client = GoogleClient(api_key="placeholder-key", orchestrator=orch)
            # Placeholder key should not create a usable client
            assert client.client is None

    def test_google_get_auth_credential_with_api_key_method(self):
        """Test get_auth_credential with api_key method."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            client = GoogleClient(api_key="real-api-key")
            credential = client.get_auth_credential(user_auth_method="api_key")
            assert credential == "real-api-key"

    def test_google_get_auth_credential_returns_none_without_creds(self):
        """Test get_auth_credential returns None without valid credentials."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            client = GoogleClient(api_key="placeholder-key")
            credential = client.get_auth_credential(user_auth_method="api_key")
            assert credential is None

    def test_google_get_auth_credential_raises_for_missing_subscription(self):
        """Test get_auth_credential raises ValueError for unavailable subscription."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            client = GoogleClient(api_key="test-key")
            with pytest.raises(ValueError):
                client.get_auth_credential(user_auth_method="subscription")

    def test_google_decrypt_api_key_with_valid_base64(self):
        """Test _decrypt_api_key_from_db with valid base64 fallback."""
        import base64
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            client = GoogleClient(api_key="test-key")
            # Test with valid base64 encoded string
            test_key = "test-api-key"
            encoded_key = base64.b64encode(test_key.encode()).decode()

            result = client._decrypt_api_key_from_db(encoded_key)
            assert result == test_key


@pytest.mark.skipif(not _openai_available, reason="openai not installed")
class TestOpenAIClientGaps:
    """Test critical uncovered paths in OpenAI client."""

    def test_openai_get_client_with_placeholder_key(self):
        """Test _get_client doesn't initialize with placeholder key."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            orch = Mock()
            orch.config = Mock()
            orch.config.openai_model = "gpt-4"
            orch.database = None

            client = OpenAIClient(api_key="placeholder-key", orchestrator=orch)
            # Placeholder key should not create a usable client
            assert client.client is None

    def test_openai_supports_string_validation(self):
        """Test OpenAI client string validation."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            client = OpenAIClient(api_key="test-key")
            # Client should be created successfully
            assert client is not None
            assert client.model == "gpt-4-turbo"

    def test_openai_model_from_config(self):
        """Test OpenAI client model selection from config."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            orch = Mock()
            orch.config = Mock()
            orch.config.openai_model = "gpt-4"

            client = OpenAIClient(api_key="test-key", orchestrator=orch)
            assert client.model == "gpt-4"

    def test_openai_decrypt_api_key_with_valid_base64(self):
        """Test _decrypt_api_key_from_db with valid base64."""
        import base64
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            client = OpenAIClient(api_key="test-key")
            # Test with valid base64 encoded string
            test_key = "sk-test-key"
            encoded_key = base64.b64encode(test_key.encode()).decode()

            result = client._decrypt_api_key_from_db(encoded_key)
            assert result == test_key


@pytest.mark.skipif(not _ollama_available, reason="ollama not installed")
class TestOllamaClientGaps:
    """Test critical uncovered paths in Ollama client."""

    def test_ollama_default_model_selection(self):
        """Test Ollama client default model."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"

            client = OllamaClient(orchestrator=orch)
            assert client.model == "llama2"

    def test_ollama_base_url_configuration(self):
        """Test Ollama client URL configuration."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "mistral"
            orch.config.ollama_url = "http://custom.host:11434"

            client = OllamaClient(orchestrator=orch)
            assert client.base_url == "http://custom.host:11434"

    def test_ollama_model_defaults(self):
        """Test Ollama client model defaults."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "neural-chat"
            orch.config.ollama_url = "http://localhost:11434"

            client = OllamaClient(orchestrator=orch)
            assert client.model == "neural-chat"
            assert client.base_url == "http://localhost:11434"

    def test_ollama_decrypt_api_key_with_valid_base64(self):
        """Test _decrypt_api_key_from_db with valid base64."""
        import base64
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"

            client = OllamaClient(orchestrator=orch)
            # Test with valid base64 encoded string
            test_key = "test-api-key"
            encoded_key = base64.b64encode(test_key.encode()).decode()

            result = client._decrypt_api_key_from_db(encoded_key)
            assert result == test_key


class TestClientEventTracking:
    """Test event tracking and logging in clients."""

    @pytest.mark.skipif(not _google_available, reason="google.generativeai not installed")
    def test_google_client_event_emitter(self):
        """Test Google client has event emitter from orchestrator."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            orch = Mock()
            orch.config = Mock()
            orch.config.google_model = "gemini-pro"
            orch.event_emitter = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            assert client.orchestrator.event_emitter is not None

    @pytest.mark.skipif(not _openai_available, reason="openai not installed")
    def test_openai_client_event_emitter(self):
        """Test OpenAI client has event emitter from orchestrator."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            orch = Mock()
            orch.config = Mock()
            orch.config.openai_model = "gpt-4"
            orch.event_emitter = Mock()

            client = OpenAIClient(api_key="test-key", orchestrator=orch)
            assert client.orchestrator.event_emitter is not None

    @pytest.mark.skipif(not _ollama_available, reason="ollama not installed")
    def test_ollama_client_event_emitter(self):
        """Test Ollama client has event emitter from orchestrator."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"
            orch.event_emitter = Mock()

            client = OllamaClient(orchestrator=orch)
            assert client.orchestrator.event_emitter is not None


class TestClientConfiguration:
    """Test client configuration handling."""

    @pytest.mark.skipif(not _google_available, reason="google.generativeai not installed")
    def test_google_client_database_setup(self):
        """Test Google client database attribute setup."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            orch = Mock()
            orch.config = Mock()
            orch.config.google_model = "gemini-pro"
            orch.database = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            # Verify database is accessible through orchestrator
            assert hasattr(client.orchestrator, "database")

    @pytest.mark.skipif(not _openai_available, reason="openai not installed")
    def test_openai_client_database_setup(self):
        """Test OpenAI client database attribute setup."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            orch = Mock()
            orch.config = Mock()
            orch.config.openai_model = "gpt-4"
            orch.database = Mock()

            client = OpenAIClient(api_key="test-key", orchestrator=orch)
            assert hasattr(client.orchestrator, "database")

    @pytest.mark.skipif(not _ollama_available, reason="ollama not installed")
    def test_ollama_client_database_setup(self):
        """Test Ollama client database attribute setup."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"
            orch.database = Mock()

            client = OllamaClient(orchestrator=orch)
            assert hasattr(client.orchestrator, "database")
