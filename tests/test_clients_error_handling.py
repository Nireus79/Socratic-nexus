"""
Initialization and validation tests for all LLM clients.
Tests basic setup, API key handling, and model selection.
"""

import importlib.util
import pytest
from unittest.mock import Mock, patch

# Check which clients are available
_google_available = importlib.util.find_spec("google.generativeai") is not None
_openai_available = importlib.util.find_spec("openai") is not None
_ollama_available = importlib.util.find_spec("ollama") is not None


@pytest.mark.skipif(not _google_available, reason="google.generativeai not installed")
class TestGoogleClientInitialization:
    """Test Google client initialization and configuration."""

    def test_google_client_missing_genai_raises_error(self):
        """Test that GoogleClient raises helpful error when google.generativeai is missing."""
        with patch("socratic_nexus.clients.google_client.genai", None):
            from socratic_nexus.clients.google_client import GoogleClient

            with pytest.raises(ImportError):
                GoogleClient(api_key="test-key")

    def test_google_placeholder_api_key_handling(self):
        """Test Google client handling of placeholder API keys."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            client = GoogleClient(api_key="placeholder-key")
            assert client.api_key == "placeholder-key"

    def test_google_subscription_token_handling(self):
        """Test Google client with subscription token."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            client = GoogleClient(api_key="test-key", subscription_token="sub-token")
            assert client.subscription_token == "sub-token"

    def test_google_model_selection_from_orchestrator(self):
        """Test Google client selects model from orchestrator."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            orch = Mock()
            orch.config = Mock()
            orch.config.google_model = "gemini-1.5-pro"

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            assert client.model == "gemini-1.5-pro"

    def test_google_default_model_when_no_orchestrator(self):
        """Test Google client uses default model when no orchestrator."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            client = GoogleClient(api_key="test-key")
            assert client.model == "gemini-pro"


@pytest.mark.skipif(not _openai_available, reason="openai not installed")
class TestOpenAIClientInitialization:
    """Test OpenAI client initialization and configuration."""

    def test_openai_placeholder_api_key_handling(self):
        """Test OpenAI client handling of placeholder API keys."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            client = OpenAIClient(api_key="placeholder-key")
            assert client.api_key == "placeholder-key"

    def test_openai_model_selection_from_orchestrator(self):
        """Test OpenAI client selects model from orchestrator."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            orch = Mock()
            orch.config = Mock()
            orch.config.openai_model = "gpt-4-turbo"

            client = OpenAIClient(api_key="test-key", orchestrator=orch)
            assert client.model == "gpt-4-turbo"

    def test_openai_default_model_when_no_orchestrator(self):
        """Test OpenAI client uses default model when no orchestrator."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            client = OpenAIClient(api_key="test-key")
            assert client.model == "gpt-4"

    def test_openai_cache_initialization(self):
        """Test OpenAI client initializes cache."""
        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            from socratic_nexus.clients.openai_client import OpenAIClient

            client = OpenAIClient(api_key="test-key")
            assert hasattr(client, "_insights_cache")
            assert isinstance(client._insights_cache, dict)


@pytest.mark.skipif(not _ollama_available, reason="ollama not installed")
class TestOllamaClientInitialization:
    """Test Ollama client initialization and configuration."""

    def test_ollama_no_api_key_required(self):
        """Test that Ollama doesn't require API key."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"

            client = OllamaClient(orchestrator=orch)
            assert client is not None

    def test_ollama_model_selection_from_orchestrator(self):
        """Test Ollama client selects model from orchestrator."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "mistral"
            orch.config.ollama_url = "http://localhost:11434"

            client = OllamaClient(orchestrator=orch)
            assert client.model == "mistral"

    def test_ollama_cache_initialization(self):
        """Test Ollama client initializes cache."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://localhost:11434"

            client = OllamaClient(orchestrator=orch)
            assert hasattr(client, "_insights_cache")
            assert isinstance(client._insights_cache, dict)

    def test_ollama_url_from_orchestrator(self):
        """Test Ollama client gets URL from orchestrator."""
        with patch("socratic_nexus.clients.ollama_client.requests.Session"):
            from socratic_nexus.clients.ollama_client import OllamaClient

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "llama2"
            orch.config.ollama_url = "http://custom-host:11434"

            client = OllamaClient(orchestrator=orch)
            assert client.base_url == "http://custom-host:11434"
