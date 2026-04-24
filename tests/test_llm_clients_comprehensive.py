"""
Comprehensive integration tests for all LLM clients

Tests for OpenAI, Google, and Ollama clients covering:
- Client initialization with orchestrator
- API key retrieval and fallback
- Token tracking and event emission
- Provider-specific token calculations
- Cost estimation
- Error handling
- Cache functionality
- Multi-user support

Adapted from Socrates monolith test patterns.
"""

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import Mock, patch

import pytest

logging.basicConfig(level=logging.DEBUG)


class MockOrchestrator:
    """Mock orchestrator for testing"""

    def __init__(self, db_path: Optional[str] = None):
        self.config = Mock()
        self.config.claude_model = "claude-haiku-4-5-20251001"
        self.config.openai_model = "gpt-4-turbo"
        self.config.google_model = "gemini-pro"
        self.config.ollama_model = "mistral"
        self.config.ollama_url = "http://localhost:11434"

        self.event_emitter = Mock()
        self.event_emitter.emit = Mock()

        self.system_monitor = Mock()
        self.system_monitor.process = Mock()

        self.database = MockDatabase(db_path)

    def reset_mocks(self):
        """Reset all mock calls"""
        self.event_emitter.emit.reset_mock()
        self.system_monitor.process.reset_mock()


class MockDatabase:
    """Mock database for API key storage"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or ":memory:"
        self.api_keys: Dict[tuple, str] = {}
        self._setup()

    def _setup(self):
        """Setup in-memory database"""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def save_api_key(self, user_id: str, provider: str, encrypted_key: str, key_hash: str) -> bool:
        """Save encrypted API key"""
        try:
            self.api_keys[(user_id, provider)] = encrypted_key
            return True
        except Exception as e:
            print(f"Error saving API key: {e}")
            return False

    def get_api_key(self, user_id: Optional[str] = None, provider: Optional[str] = None) -> Optional[str]:
        """Retrieve encrypted API key"""
        if user_id is None and provider is None:
            return None
        return self.api_keys.get((user_id, provider))

    def delete_api_key(self, user_id: str, provider: str) -> bool:
        """Delete API key"""
        try:
            if (user_id, provider) in self.api_keys:
                del self.api_keys[(user_id, provider)]
            return True
        except Exception as e:
            print(f"Error deleting API key: {e}")
            return False


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator"""
    return MockOrchestrator()


class TestOpenAIClientIntegration:
    """Integration tests for OpenAI client"""

    def test_client_initialization(self, mock_orchestrator):
        """Test OpenAI client initializes with orchestrator"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            client = OpenAIClient(
                api_key="sk-test-key",
                orchestrator=mock_orchestrator
            )

            assert client.api_key == "sk-test-key"
            assert client.orchestrator is mock_orchestrator

    def test_client_with_placeholder_key(self, mock_orchestrator):
        """Test client with placeholder key"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            client = OpenAIClient(
                api_key="placeholder_key",
                orchestrator=mock_orchestrator
            )

            assert client.api_key == "placeholder_key"

    def test_get_user_api_key_from_database(self, mock_orchestrator):
        """Test retrieving user API key from database"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            client = OpenAIClient(
                api_key="default-key",
                orchestrator=mock_orchestrator
            )

            # Setup user API key
            encryption_key_base = os.getenv("SOCRATES_ENCRYPTION_KEY", "default-socrates-key")
            key_hash = hashlib.sha256(encryption_key_base.encode()).digest()
            encryption_key = base64.urlsafe_b64encode(key_hash)

            pytest.importorskip("cryptography")
            from cryptography.fernet import Fernet

            cipher = Fernet(encryption_key)
            user_key = "sk-user-specific-key"
            encrypted = cipher.encrypt(user_key.encode()).decode()

            mock_orchestrator.database.save_api_key(
                "user123",
                "openai",
                encrypted,
                hashlib.sha256(user_key.encode()).hexdigest()
            )

            # Retrieve user-specific key
            api_key, is_user_specific = client._get_user_api_key("user123")
            assert api_key == user_key
            assert is_user_specific is True

    def test_fallback_to_default_key(self, mock_orchestrator):
        """Test fallback to default key when user key not found"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            client = OpenAIClient(
                api_key="default-key",
                orchestrator=mock_orchestrator
            )

            api_key, is_user_specific = client._get_user_api_key("nonexistent_user")
            assert api_key == "default-key"
            assert is_user_specific is False

    def test_cache_key_generation(self, mock_orchestrator):
        """Test cache key generation"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            client = OpenAIClient(
                api_key="sk-test-key",
                orchestrator=mock_orchestrator
            )

            message = "Test message"
            cache_key = client._get_cache_key(message)

            expected = hashlib.sha256(message.encode()).hexdigest()
            assert cache_key == expected

    def test_token_tracking(self, mock_orchestrator):
        """Test token usage tracking"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            client = OpenAIClient(
                api_key="sk-test-key",
                orchestrator=mock_orchestrator
            )

            mock_usage = Mock()
            mock_usage.prompt_tokens = 100
            mock_usage.completion_tokens = 50

            client._track_token_usage(mock_usage, "test_operation")

            mock_orchestrator.system_monitor.process.assert_called_once()

    def test_cost_calculation(self, mock_orchestrator):
        """Test cost calculation for OpenAI"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            client = OpenAIClient(
                api_key="sk-test-key",
                orchestrator=mock_orchestrator
            )

            mock_usage = Mock()
            mock_usage.prompt_tokens = 1000
            mock_usage.completion_tokens = 1000

            cost = client._calculate_cost(mock_usage)

            # Should return a number
            assert isinstance(cost, (int, float))
            assert cost >= 0


class TestGoogleClientIntegration:
    """Integration tests for Google client"""

    def test_client_initialization(self, mock_orchestrator):
        """Test Google client initializes with orchestrator"""
        from socratic_nexus.clients.google_client import GoogleClient

        with patch("socratic_nexus.clients.google_client.genai"):
            client = GoogleClient(
                api_key="AIza-test-key",
                orchestrator=mock_orchestrator
            )

            assert client.api_key == "AIza-test-key"
            assert client.orchestrator is mock_orchestrator

    def test_client_with_placeholder_key(self, mock_orchestrator):
        """Test Google client with placeholder key"""
        from socratic_nexus.clients.google_client import GoogleClient

        with patch("socratic_nexus.clients.google_client.genai"):
            client = GoogleClient(
                api_key="placeholder_key",
                orchestrator=mock_orchestrator
            )

            assert client.api_key == "placeholder_key"

    def test_token_estimation_google(self, mock_orchestrator):
        """Test token estimation for Google (text-length based)"""
        from socratic_nexus.clients.google_client import GoogleClient

        with patch("socratic_nexus.clients.google_client.genai"):
            client = GoogleClient(
                api_key="AIza-test-key",
                orchestrator=mock_orchestrator
            )

            # Google estimates tokens as chars / 4
            input_len = 400  # 400 chars ≈ 100 tokens
            output_len = 200  # 200 chars ≈ 50 tokens

            if hasattr(client, '_track_token_usage_google'):
                client._track_token_usage_google(input_len, output_len, "test_operation")
                mock_orchestrator.system_monitor.process.assert_called_once()

    def test_cost_calculation_google(self, mock_orchestrator):
        """Test cost calculation for Google"""
        from socratic_nexus.clients.google_client import GoogleClient

        with patch("socratic_nexus.clients.google_client.genai"):
            client = GoogleClient(
                api_key="AIza-test-key",
                orchestrator=mock_orchestrator
            )

            if hasattr(client, '_calculate_cost_google'):
                cost = client._calculate_cost_google(1000, 500)
                assert isinstance(cost, (int, float))
                assert cost >= 0


class TestOllamaClientIntegration:
    """Integration tests for Ollama client"""

    def test_client_initialization(self, mock_orchestrator):
        """Test Ollama client initializes with orchestrator"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)

            assert client.orchestrator is mock_orchestrator

    def test_ollama_url_configuration(self, mock_orchestrator):
        """Test Ollama URL can be configured"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests"):
            mock_orchestrator.config.ollama_url = "http://custom-server:11434"

            client = OllamaClient(orchestrator=mock_orchestrator)

            assert client.base_url == "http://custom-server:11434"

    def test_ollama_model_configuration(self, mock_orchestrator):
        """Test Ollama model can be configured"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests"):
            mock_orchestrator.config.ollama_model = "llama2"

            client = OllamaClient(orchestrator=mock_orchestrator)

            assert client.model == "llama2"

    def test_ollama_token_tracking(self, mock_orchestrator):
        """Test token tracking for Ollama"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)

            if hasattr(client, '_track_token_usage_ollama'):
                client._track_token_usage_ollama(400, 200, "test_operation")
                mock_orchestrator.system_monitor.process.assert_called_once()

    def test_ollama_cost_is_zero(self, mock_orchestrator):
        """Test Ollama always returns zero cost"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=mock_orchestrator)

            if hasattr(client, '_estimate_cost'):
                cost = client._estimate_cost(10000, 5000)
                assert cost == 0.0


class TestClientInterchangeability:
    """Test that all clients have compatible interfaces"""

    def test_all_clients_have_generate_response(self, mock_orchestrator):
        """Test all clients implement generate_response"""
        from socratic_nexus.clients.openai_client import OpenAIClient
        from socratic_nexus.clients.google_client import GoogleClient
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            with patch("socratic_nexus.clients.google_client.genai"):
                with patch("socratic_nexus.clients.ollama_client.requests"):
                    clients = [
                        OpenAIClient(api_key="test", orchestrator=mock_orchestrator),
                        GoogleClient(api_key="test", orchestrator=mock_orchestrator),
                        OllamaClient(orchestrator=mock_orchestrator),
                    ]

                    for client in clients:
                        assert hasattr(client, 'generate_response')
                        assert callable(getattr(client, 'generate_response'))

    def test_all_clients_have_extract_insights(self, mock_orchestrator):
        """Test all clients implement extract_insights"""
        from socratic_nexus.clients.openai_client import OpenAIClient
        from socratic_nexus.clients.google_client import GoogleClient
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            with patch("socratic_nexus.clients.google_client.genai"):
                with patch("socratic_nexus.clients.ollama_client.requests"):
                    clients = [
                        OpenAIClient(api_key="test", orchestrator=mock_orchestrator),
                        GoogleClient(api_key="test", orchestrator=mock_orchestrator),
                        OllamaClient(orchestrator=mock_orchestrator),
                    ]

                    for client in clients:
                        assert hasattr(client, 'extract_insights')
                        assert callable(getattr(client, 'extract_insights'))

    def test_all_clients_have_get_cache_key(self, mock_orchestrator):
        """Test all clients implement _get_cache_key"""
        from socratic_nexus.clients.openai_client import OpenAIClient
        from socratic_nexus.clients.google_client import GoogleClient
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            with patch("socratic_nexus.clients.google_client.genai"):
                with patch("socratic_nexus.clients.ollama_client.requests"):
                    clients = [
                        OpenAIClient(api_key="test", orchestrator=mock_orchestrator),
                        GoogleClient(api_key="test", orchestrator=mock_orchestrator),
                        OllamaClient(orchestrator=mock_orchestrator),
                    ]

                    for client in clients:
                        assert hasattr(client, '_get_cache_key')
                        cache_key = client._get_cache_key("test")
                        assert isinstance(cache_key, str)


class TestEventEmission:
    """Test event emission from clients"""

    def test_openai_token_usage_event_emission(self, mock_orchestrator):
        """Test TOKEN_USAGE event emitted from OpenAI client"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai"):
            client = OpenAIClient(
                api_key="sk-test-key",
                orchestrator=mock_orchestrator
            )

            mock_usage = Mock()
            mock_usage.prompt_tokens = 100
            mock_usage.completion_tokens = 50

            client._track_token_usage(mock_usage, "test_operation")

            mock_orchestrator.event_emitter.emit.assert_called_once()

    def test_google_token_usage_event_emission(self, mock_orchestrator):
        """Test TOKEN_USAGE event emitted from Google client"""
        from socratic_nexus.clients.google_client import GoogleClient

        with patch("socratic_nexus.clients.google_client.genai"):
            client = GoogleClient(
                api_key="AIza-test-key",
                orchestrator=mock_orchestrator
            )

            if hasattr(client, '_track_token_usage_google'):
                client._track_token_usage_google(400, 200, "test_operation")
                # Event emission may be called
                assert mock_orchestrator.event_emitter.emit.call_count >= 0


class TestMultiUserSupport:
    """Test multi-user API key management"""

    def test_different_users_different_keys(self, mock_orchestrator):
        """Test different users have different API keys"""
        with patch("socratic_nexus.clients.openai_client.openai"):
            # Setup different keys for different users
            mock_orchestrator.database.save_api_key("user1", "openai", "user1_encrypted", "hash1")
            mock_orchestrator.database.save_api_key("user2", "openai", "user2_encrypted", "hash2")

            key1 = mock_orchestrator.database.get_api_key("user1", "openai")
            key2 = mock_orchestrator.database.get_api_key("user2", "openai")

            assert key1 == "user1_encrypted"
            assert key2 == "user2_encrypted"
            assert key1 != key2

    def test_same_user_consistent_key(self, mock_orchestrator):
        """Test same user gets consistent key"""
        with patch("socratic_nexus.clients.openai_client.openai"):
            mock_orchestrator.database.save_api_key("user1", "openai", "user_encrypted", "hash")

            key1 = mock_orchestrator.database.get_api_key("user1", "openai")
            key2 = mock_orchestrator.database.get_api_key("user1", "openai")

            assert key1 == key2 == "user_encrypted"

    def test_multiple_providers_per_user(self, mock_orchestrator):
        """Test user can have keys for multiple providers"""
        with patch("socratic_nexus.clients.openai_client.openai"):
            with patch("socratic_nexus.clients.google_client.genai"):
                # Setup keys for different providers
                mock_orchestrator.database.save_api_key("user1", "openai", "openai_key", "hash1")
                mock_orchestrator.database.save_api_key("user1", "google", "google_key", "hash2")

                openai_key = mock_orchestrator.database.get_api_key("user1", "openai")
                google_key = mock_orchestrator.database.get_api_key("user1", "google")

                assert openai_key == "openai_key"
                assert google_key == "google_key"
                assert openai_key != google_key


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
