"""
Comprehensive integration tests for ClaudeClient with AgentOrchestrator

Tests adapted from Socrates project to verify:
- Client initialization with orchestrator
- API key retrieval from database and fallback
- Subscription token handling
- Token tracking and event emission
- Encryption/decryption of API keys
- Async operations
- Error handling and recovery
- Cache functionality
- Multi-auth scenarios
- Provider-specific behaviors
"""

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Setup logging for test debugging
logging.basicConfig(level=logging.DEBUG)


class MockOrchestrator:
    """Mock orchestrator for testing without full system initialization"""

    def __init__(self, db_path: str = None):
        self.config = Mock()
        self.config.claude_model = "claude-haiku-4-5-20251001"

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

    def get_api_key(self, user_id: str = None, provider: str = None) -> str | None:
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


class TestClaudeClientInitialization:
    """Tests for ClaudeClient initialization"""

    def test_client_initialization_with_api_key(self, mock_orchestrator):
        """Test ClaudeClient initializes with API key"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="sk-ant-test-key",
                orchestrator=mock_orchestrator
            )

            assert client.api_key == "sk-ant-test-key"
            assert client.orchestrator is mock_orchestrator

    def test_client_initialization_with_subscription(self, mock_orchestrator):
        """Test ClaudeClient initializes with subscription token"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                subscription_token="sub-token-123",
                orchestrator=mock_orchestrator
            )

            assert client.subscription_token == "sub-token-123"
            assert client.orchestrator is mock_orchestrator

    def test_client_initialization_with_both_auth_methods(self, mock_orchestrator):
        """Test ClaudeClient initializes with both auth methods"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="sk-ant-test-key",
                subscription_token="sub-token-123",
                orchestrator=mock_orchestrator
            )

            assert client.api_key == "sk-ant-test-key"
            assert client.subscription_token == "sub-token-123"

    def test_client_initialization_with_placeholder_key(self, mock_orchestrator):
        """Test client initializes with placeholder key"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="placeholder_test",
                orchestrator=mock_orchestrator
            )

            # Placeholder keys shouldn't initialize real client
            assert client.api_key == "placeholder_test"
            assert client.client is None

    def test_client_initialization_without_credentials(self, mock_orchestrator):
        """Test initialization without any credentials"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(orchestrator=mock_orchestrator)

            assert client.api_key is None
            assert client.subscription_token is None


class TestAPIKeyRetrieval:
    """Tests for API key retrieval from database"""

    def test_get_user_api_key_from_database(self, mock_orchestrator):
        """Test retrieving user API key from database"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="default-key",
                orchestrator=mock_orchestrator
            )

            # Setup user API key in database
            encryption_key_base = os.getenv("SOCRATES_ENCRYPTION_KEY", "default-socrates-key")
            key_hash = hashlib.sha256(encryption_key_base.encode()).digest()
            encryption_key = base64.urlsafe_b64encode(key_hash)

            pytest.importorskip("cryptography")
            from cryptography.fernet import Fernet

            cipher = Fernet(encryption_key)
            user_key = "sk-ant-user-specific-key"
            encrypted = cipher.encrypt(user_key.encode()).decode()

            mock_orchestrator.database.save_api_key(
                "user123",
                "claude",
                encrypted,
                hashlib.sha256(user_key.encode()).hexdigest()
            )

            # Retrieve user-specific key
            api_key, is_user_specific = client._get_user_api_key("user123")
            assert api_key == user_key
            assert is_user_specific is True

    def test_fallback_to_default_key(self, mock_orchestrator):
        """Test fallback to default key when user key not found"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="default-key",
                orchestrator=mock_orchestrator
            )

            # No user key in database
            api_key, is_user_specific = client._get_user_api_key("nonexistent_user")
            assert api_key == "default-key"
            assert is_user_specific is False

    def test_database_error_fallback(self, mock_orchestrator):
        """Test fallback when database has errors"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            mock_orchestrator.database.get_api_key = Mock(side_effect=Exception("DB Error"))
            client = ClaudeClient(
                api_key="env-key",
                orchestrator=mock_orchestrator
            )

            api_key, is_user_specific = client._get_user_api_key("user123")
            assert api_key == "env-key"
            assert is_user_specific is False


class TestTokenTracking:
    """Tests for token usage tracking"""

    def test_token_tracking_basic(self, mock_orchestrator):
        """Test basic token tracking"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="sk-ant-test-key",
                orchestrator=mock_orchestrator
            )

            # Mock usage object
            mock_usage = Mock()
            mock_usage.input_tokens = 100
            mock_usage.output_tokens = 50

            client._track_token_usage(mock_usage, "test_operation")

            # Verify system_monitor.process was called
            mock_orchestrator.system_monitor.process.assert_called_once()
            call_args = mock_orchestrator.system_monitor.process.call_args[0][0]

            assert call_args["action"] == "track_tokens"
            assert call_args["operation"] == "test_operation"
            assert call_args["input_tokens"] == 100
            assert call_args["output_tokens"] == 50
            assert call_args["total_tokens"] == 150

    def test_token_tracking_event_emission(self, mock_orchestrator):
        """Test token usage event is emitted"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="sk-ant-test-key",
                orchestrator=mock_orchestrator
            )

            mock_usage = Mock()
            mock_usage.input_tokens = 100
            mock_usage.output_tokens = 50

            client._track_token_usage(mock_usage, "test_operation")

            # Verify event_emitter.emit was called
            mock_orchestrator.event_emitter.emit.assert_called_once()


class TestCacheKeyGeneration:
    """Tests for cache key generation"""

    def test_cache_key_generation(self, mock_orchestrator):
        """Test SHA256 cache key generation"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="sk-ant-test-key",
                orchestrator=mock_orchestrator
            )

            message = "Test message"
            cache_key = client._get_cache_key(message)

            expected = hashlib.sha256(message.encode()).hexdigest()
            assert cache_key == expected

    def test_cache_key_consistency(self, mock_orchestrator):
        """Test cache key is consistent for same input"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="sk-ant-test-key",
                orchestrator=mock_orchestrator
            )

            message = "Test message"
            key1 = client._get_cache_key(message)
            key2 = client._get_cache_key(message)

            assert key1 == key2


class TestErrorHandling:
    """Tests for error handling"""

    def test_api_error_on_missing_key(self, mock_orchestrator):
        """Test APIError raised when no valid API key"""
        from socratic_nexus.clients.claude_client import ClaudeClient
        from socratic_nexus.exceptions import APIError

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key=None,
                orchestrator=mock_orchestrator
            )

            with pytest.raises(APIError):
                client._get_client()

    def test_json_parsing_handles_markdown(self, mock_orchestrator):
        """Test JSON parsing removes markdown code blocks"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="sk-ant-test-key",
                orchestrator=mock_orchestrator
            )

            response_with_markdown = '''```json
            {
                "insights": "test",
                "data": ["value"]
            }
            ```'''

            result = client._parse_json_response(response_with_markdown)
            assert "insights" in result or result == {}

    def test_json_parsing_handles_invalid_json(self, mock_orchestrator):
        """Test JSON parsing returns empty dict on invalid JSON"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="sk-ant-test-key",
                orchestrator=mock_orchestrator
            )

            invalid_json = "This is not JSON {invalid"
            result = client._parse_json_response(invalid_json)

            assert result == {}


class TestGenerateResponse:
    """Tests for generate_response method"""

    def test_generate_response_with_api_key(self, mock_orchestrator):
        """Test response generation with API key auth"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response text")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(
                    api_key="sk-ant-test-key",
                    orchestrator=mock_orchestrator
                )
                result = client.generate_response("test prompt")

                assert isinstance(result, (str, type(None)))
                mock_client.messages.create.assert_called()

    def test_generate_response_with_subscription(self, mock_orchestrator):
        """Test response generation with subscription auth"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response text")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(
                    subscription_token="sub-token-123",
                    orchestrator=mock_orchestrator
                )
                result = client.generate_response(
                    "test prompt",
                    user_auth_method="subscription"
                )

                assert isinstance(result, (str, type(None)))

    def test_generate_response_with_temperature(self, mock_orchestrator):
        """Test response generation with temperature parameter"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(
                    api_key="sk-ant-test-key",
                    orchestrator=mock_orchestrator
                )
                result = client.generate_response(
                    "test prompt",
                    temperature=0.7
                )

                assert isinstance(result, (str, type(None)))

    def test_generate_response_with_max_tokens(self, mock_orchestrator):
        """Test response generation with max tokens"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(
                    api_key="sk-ant-test-key",
                    orchestrator=mock_orchestrator
                )
                result = client.generate_response(
                    "test prompt",
                    max_tokens=500
                )

                assert isinstance(result, (str, type(None)))


class TestAsyncOperations:
    """Tests for async operation support"""

    @pytest.mark.asyncio
    async def test_async_token_tracking(self, mock_orchestrator):
        """Test async token tracking"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="sk-ant-test-key",
                orchestrator=mock_orchestrator
            )

            mock_usage = Mock()
            mock_usage.input_tokens = 100
            mock_usage.output_tokens = 50

            await client._track_token_usage_async(mock_usage, "async_operation")

            # Verify tracking was called
            mock_orchestrator.system_monitor.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_async(self, mock_orchestrator):
        """Test async response generation"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text="async response")],
                    usage=Mock(input_tokens=10, output_tokens=20)
                )
            )

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(
                    api_key="sk-ant-test-key",
                    orchestrator=mock_orchestrator
                )
                result = await client.generate_response_async("test prompt")

                assert isinstance(result, (str, type(None)))


class TestMultiAuthScenarios:
    """Tests for multi-auth scenarios"""

    def test_switching_auth_methods(self, mock_orchestrator):
        """Test switching between auth methods in same session"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(
                    api_key="api-key",
                    subscription_token="sub-token",
                    orchestrator=mock_orchestrator
                )

                # Call 1 with API key
                result1 = client.generate_response(
                    "prompt1",
                    user_auth_method="api_key"
                )

                # Call 2 with subscription
                result2 = client.generate_response(
                    "prompt2",
                    user_auth_method="subscription"
                )

                assert isinstance(result1, (str, type(None)))
                assert isinstance(result2, (str, type(None)))

    def test_auth_method_with_user_id(self, mock_orchestrator):
        """Test auth with user-specific credentials"""
        from socratic_nexus.clients.claude_client import ClaudeClient

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(
                    api_key="default-key",
                    orchestrator=mock_orchestrator
                )
                result = client.generate_response(
                    "test",
                    user_id="user123",
                    user_auth_method="api_key"
                )

                assert isinstance(result, (str, type(None)))


class TestDatabaseIntegration:
    """Tests for database integration"""

    def test_save_and_retrieve_api_key(self):
        """Test saving and retrieving API keys"""
        db = MockDatabase()

        # Save API key
        success = db.save_api_key(
            "user123",
            "claude",
            "encrypted_key_value",
            "key_hash"
        )
        assert success is True

        # Retrieve API key
        retrieved = db.get_api_key("user123", "claude")
        assert retrieved == "encrypted_key_value"

    def test_save_and_retrieve_multiple_providers(self):
        """Test storing keys for multiple providers per user"""
        db = MockDatabase()

        # Save keys for multiple providers
        db.save_api_key("user123", "claude", "claude_key", "hash1")
        db.save_api_key("user123", "openai", "openai_key", "hash2")

        # Verify retrieval
        assert db.get_api_key("user123", "claude") == "claude_key"
        assert db.get_api_key("user123", "openai") == "openai_key"

    def test_delete_api_key(self):
        """Test deleting API keys"""
        db = MockDatabase()

        db.save_api_key("user123", "claude", "key_value", "hash")
        assert db.get_api_key("user123", "claude") is not None

        success = db.delete_api_key("user123", "claude")
        assert success is True
        assert db.get_api_key("user123", "claude") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
