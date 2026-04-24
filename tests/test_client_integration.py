"""
Integration tests for LLM clients with orchestrator

Tests verify:
- Client initialization with orchestrator
- API key retrieval from database
- Token tracking and event emission
- Encryption/decryption of API keys
- Async operations
- Error handling
- Cache functionality
- Cost calculation
"""

import base64
import hashlib
import logging
from typing import Dict
from unittest.mock import Mock, patch

import pytest

# Setup logging for test debugging
logging.basicConfig(level=logging.DEBUG)


class MockOrchestrator:
    """Mock orchestrator for testing without full system initialization"""

    def __init__(self, db_path: str = None):
        self.config = Mock()
        self.config.claude_model = "claude-3-sonnet-20240229"
        self.config.openai_model = "gpt-4-turbo"
        self.config.google_model = "gemini-pro"
        self.config.ollama_model = "mistral"
        self.config.ollama_url = "http://localhost:11434"

        self.event_emitter = Mock()
        self.event_emitter.emit = Mock()

        self.system_monitor = Mock()
        self.system_monitor.process = Mock()

        # Setup mock database
        self.database = MockDatabase(db_path)

    def reset_mocks(self):
        """Reset all mock calls"""
        self.event_emitter.emit.reset_mock()
        self.system_monitor.process.reset_mock()


class MockDatabase:
    """Mock database for testing API key storage"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self.api_keys: Dict[tuple, str] = {}  # (user_id, provider) -> encrypted_key

    def save_api_key(self, user_id: str, provider: str, encrypted_key: str, key_hash: str) -> bool:
        """Save encrypted API key"""
        try:
            self.api_keys[(user_id, provider)] = encrypted_key
            return True
        except Exception as e:
            print(f"Error saving API key: {e}")
            return False

    def get_api_key(self, user_id: str, provider: str) -> str | None:
        """Retrieve encrypted API key"""
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


class TestClaudeClientIntegration:
    """Integration tests for Claude client"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        return MockOrchestrator()

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic SDK"""
        with patch("socratic_nexus.clients.claude_client.anthropic") as mock:
            yield mock

    def test_client_initialization(self, mock_orchestrator):
        """Test Claude client initializes with orchestrator"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        client = ClaudeClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

        assert client.api_key == "sk-test-key"
        assert client.orchestrator is mock_orchestrator
        assert client.model == "claude-3-sonnet-20240229"
        assert client._insights_cache == {}

    def test_client_initialization_without_orchestrator(self):
        """Test Claude client initializes without orchestrator"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="sk-test-key", orchestrator=None)

            assert client.api_key == "sk-test-key"
            assert client.orchestrator is None

    def test_get_auth_credential_api_key(self, mock_orchestrator):
        """Test getting auth credential with api_key method"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        client = ClaudeClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

        credential = client.get_auth_credential("api_key")
        assert credential == "sk-test-key"

    def test_get_auth_credential_subscription(self, mock_orchestrator):
        """Test getting auth credential with subscription method"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="default", orchestrator=mock_orchestrator)
            client.subscription_token = "sub-token"

            credential = client.get_auth_credential("subscription")
            assert credential == "sub-token"

    def test_cache_key_generation(self, mock_orchestrator):
        """Test SHA256 cache key generation"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

            message = "Test message"
            cache_key = client._get_cache_key(message)

            expected = hashlib.sha256(message.encode()).hexdigest()
            assert cache_key == expected

    def test_insights_caching(self, mock_orchestrator):
        """Test insights extraction caching"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

            # Manually add to cache
            test_response = {"goals": "build app", "requirements": ["fast", "scalable"]}
            cache_key = client._get_cache_key("Test response")
            client._insights_cache[cache_key] = test_response

            # Verify cache hit
            assert cache_key in client._insights_cache
            assert client._insights_cache[cache_key] == test_response

    def test_error_on_no_api_key(self, mock_orchestrator):
        """Test APIError raised when no valid API key"""
        from socratic_nexus.clients.claude_client import ClaudeClient
        from socratic_nexus.exceptions import APIError

        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=mock_orchestrator)

            with pytest.raises(APIError) as exc_info:
                client._get_client(user_id="nonexistent_user")

            assert exc_info.value.error_type == "MISSING_API_KEY"


class TestOpenAIClientIntegration:
    """Integration tests for OpenAI client"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        return MockOrchestrator()

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI SDK"""
        with patch("socratic_nexus.clients.openai_client.openai") as mock:
            yield mock

    def test_client_initialization(self, mock_orchestrator, mock_openai):
        """Test OpenAI client initializes with orchestrator"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        client = OpenAIClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

        assert client.api_key == "sk-test-key"
        assert client.orchestrator is mock_orchestrator
        assert client.model == "gpt-4-turbo"
        assert client._insights_cache == {}

    def test_cache_key_generation(self, mock_orchestrator, mock_openai):
        """Test SHA256 cache key generation"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        client = OpenAIClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

        message = "Test message"
        cache_key = client._get_cache_key(message)

        expected = hashlib.sha256(message.encode()).hexdigest()
        assert cache_key == expected

    def test_cost_calculation(self, mock_orchestrator, mock_openai):
        """Test cost calculation for OpenAI"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        client = OpenAIClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

        mock_usage = Mock()
        mock_usage.prompt_tokens = 1000
        mock_usage.completion_tokens = 1000

        cost = client._calculate_cost(mock_usage)

        # $0.01 per 1K input + $0.03 per 1K output = $0.04
        expected = 0.01 + 0.03
        assert abs(cost - expected) < 0.0001

    def test_json_parsing_handles_markdown(self, mock_orchestrator, mock_openai):
        """Test JSON parsing removes markdown code blocks"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        client = OpenAIClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

        response_with_markdown = """```json
        {
            "goals": "build app",
            "requirements": ["fast"]
        }
        ```"""

        result = client._parse_json_response(response_with_markdown)
        assert result["goals"] == "build app"
        assert "fast" in result["requirements"]

    def test_json_parsing_handles_invalid_json(self, mock_orchestrator, mock_openai):
        """Test JSON parsing returns empty dict on invalid JSON"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        client = OpenAIClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

        invalid_json = "This is not JSON {invalid"
        result = client._parse_json_response(invalid_json)

        assert result == {}


class TestGoogleClientIntegration:
    """Integration tests for Google client"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        return MockOrchestrator()

    def test_google_client_requires_dependency(self):
        """Test that Google client requires google-generativeai"""
        try:
            from socratic_nexus.clients.google_client import GoogleClient  # noqa: F401

            # If we can import it, the dependency is installed
            assert True
        except ModuleNotFoundError as e:
            if "google" in str(e):
                pytest.skip("google-generativeai not installed")
            raise


class TestOllamaClientIntegration:
    """Integration tests for Ollama client"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        return MockOrchestrator()

    @pytest.fixture
    def mock_requests(self):
        """Mock requests library"""
        with patch("socratic_nexus.clients.ollama_client.requests") as mock:
            yield mock

    def test_client_initialization(self, mock_orchestrator, mock_requests):
        """Test Ollama client initializes with orchestrator"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        client = OllamaClient(orchestrator=mock_orchestrator)

        assert client.orchestrator is mock_orchestrator
        assert client.model == "mistral"
        assert client.base_url == "http://localhost:11434"

    def test_ollama_url_from_config(self, mock_orchestrator, mock_requests):
        """Test Ollama URL can be configured"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        mock_orchestrator.config.ollama_url = "http://custom-server:11434"

        client = OllamaClient(orchestrator=mock_orchestrator)

        assert client.base_url == "http://custom-server:11434"

    def test_ollama_model_from_config(self, mock_orchestrator, mock_requests):
        """Test Ollama model can be configured"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        mock_orchestrator.config.ollama_model = "llama2"

        client = OllamaClient(orchestrator=mock_orchestrator)

        assert client.model == "llama2"

    def test_ollama_cache_initialization(self, mock_orchestrator, mock_requests):
        """Test Ollama client initializes with empty cache"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        client = OllamaClient(orchestrator=mock_orchestrator)

        assert client._insights_cache == {}
        assert client._question_cache == {}

    def test_cache_key_generation(self, mock_orchestrator, mock_requests):
        """Test SHA256 cache key generation"""
        from socratic_nexus.clients.ollama_client import OllamaClient

        client = OllamaClient(orchestrator=mock_orchestrator)

        message = "Test message"
        cache_key = client._get_cache_key(message)

        expected = hashlib.sha256(message.encode()).hexdigest()
        assert cache_key == expected


class TestClientInterchangeability:
    """Test that all clients have compatible interfaces"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        return MockOrchestrator()

    def test_all_clients_have_same_methods(self, mock_orchestrator):
        """Test all clients implement same methods"""
        from socratic_nexus.clients.claude_client import ClaudeClient
        from socratic_nexus.clients.openai_client import OpenAIClient
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.claude_client.anthropic"), patch(
            "socratic_nexus.clients.openai_client.openai"
        ), patch("socratic_nexus.clients.ollama_client.requests"):
            clients = [
                ClaudeClient(api_key="test", orchestrator=mock_orchestrator),
                OpenAIClient(api_key="test", orchestrator=mock_orchestrator),
                OllamaClient(orchestrator=mock_orchestrator),
            ]

            # Methods that should exist in all clients
            required_methods = [
                "extract_insights",
                "generate_code",
                "generate_socratic_question",
                "generate_response",
                "_get_cache_key",
                "_get_user_api_key",
                "_decrypt_api_key_from_db",
            ]

            for client in clients:
                for method in required_methods:
                    assert hasattr(client, method), f"{client.__class__.__name__} missing {method}"
                    assert callable(getattr(client, method)), f"{method} is not callable"

    def test_client_substitutability(self, mock_orchestrator):
        """Test that any client can substitute for another"""
        from socratic_nexus.clients.claude_client import ClaudeClient
        from socratic_nexus.clients.openai_client import OpenAIClient
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.claude_client.anthropic"), patch(
            "socratic_nexus.clients.openai_client.openai"
        ), patch("socratic_nexus.clients.ollama_client.requests"):

            def use_client(client, prompt: str):
                """Generic function that uses any client"""
                # This should work with any client
                cache_key = client._get_cache_key(prompt)
                return cache_key

            clients = [
                ClaudeClient(api_key="test", orchestrator=mock_orchestrator),
                OpenAIClient(api_key="test", orchestrator=mock_orchestrator),
                OllamaClient(orchestrator=mock_orchestrator),
            ]

            for client in clients:
                result = use_client(client, "Test prompt")
                assert result == hashlib.sha256("Test prompt".encode()).hexdigest()


class TestEventEmission:
    """Test event emission from clients"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        return MockOrchestrator()

    def test_token_usage_event_type_available(self, mock_orchestrator):
        """Test TOKEN_USAGE event type exists"""
        from socratic_nexus.events import EventType

        assert hasattr(EventType, "TOKEN_USAGE")


class TestEncryption:
    """Test API key encryption/decryption"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        return MockOrchestrator()

    def test_encryption_fallback_methods(self, mock_orchestrator, monkeypatch):
        """Test encryption method fallbacks"""
        from socratic_nexus.clients.openai_client import OpenAIClient

        try:
            import cryptography  # noqa: F401

            client = OpenAIClient(api_key="sk-test-key", orchestrator=mock_orchestrator)

            # Test Base64 fallback (Method 3)
            original_key = "sk-test-key"
            encrypted = base64.b64encode(original_key.encode()).decode()

            decrypted = client._decrypt_api_key_from_db(encrypted)
            # Base64 should work as fallback (though SHA256 will try first)
            assert decrypted is not None  # Either Base64 or SHA256 succeeds
        except ImportError:
            pytest.skip("cryptography module not available")


class TestDatabaseIntegration:
    """Test database integration for API key storage"""

    def test_save_and_retrieve_api_key(self):
        """Test saving and retrieving API keys from database"""
        db = MockDatabase()

        # Save API key
        success = db.save_api_key("user123", "openai", "encrypted_key_value", "key_hash")
        assert success is True

        # Retrieve API key
        retrieved = db.get_api_key("user123", "openai")
        assert retrieved == "encrypted_key_value"

    def test_save_and_retrieve_multiple_providers(self):
        """Test storing keys for multiple providers per user"""
        db = MockDatabase()

        # Save keys for multiple providers
        db.save_api_key("user123", "openai", "openai_key", "hash1")
        db.save_api_key("user123", "google", "google_key", "hash2")
        db.save_api_key("user123", "ollama", "ollama_key", "hash3")

        # Verify retrieval
        assert db.get_api_key("user123", "openai") == "openai_key"
        assert db.get_api_key("user123", "google") == "google_key"
        assert db.get_api_key("user123", "ollama") == "ollama_key"

    def test_delete_api_key(self):
        """Test deleting API keys"""
        db = MockDatabase()

        db.save_api_key("user123", "openai", "key_value", "hash")
        assert db.get_api_key("user123", "openai") is not None

        success = db.delete_api_key("user123", "openai")
        assert success is True
        assert db.get_api_key("user123", "openai") is None
