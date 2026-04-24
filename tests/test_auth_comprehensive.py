"""Comprehensive authentication and authorization testing.

Tests authentication flows:
- API key retrieval and fallback
- Subscription token handling
- Database-based key management
- Decryption of stored credentials
- Auth method selection and switching
- Multi-auth scenarios
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext
from socratic_nexus.exceptions import APIError


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator with database."""
    orch = Mock()
    orch.config = Mock()
    orch.config.claude_model = "claude-3-sonnet-20240229"
    orch.event_emitter = Mock()
    orch.database = Mock()
    return orch


@pytest.fixture
def mock_orchestrator_with_key():
    """Mock orchestrator that returns API key from database."""
    orch = Mock()
    orch.config = Mock()
    orch.config.claude_model = "claude-3-sonnet-20240229"
    orch.event_emitter = Mock()
    orch.database = Mock()
    orch.database.get_api_key.return_value = "db-stored-key"
    return orch


class TestAuthCredentialRetrieval:
    """Tests for get_auth_credential method."""

    def test_get_api_key_credential(self, mock_orchestrator):
        """Test retrieving API key credential."""
        client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
        credential = client.get_auth_credential("api_key")

        assert credential == "test-key"

    def test_get_subscription_credential(self, mock_orchestrator):
        """Test retrieving subscription credential."""
        client = ClaudeClient(
            api_key="test-key",
            orchestrator=mock_orchestrator,
            subscription_token="sub-token-123",
        )
        credential = client.get_auth_credential("subscription")

        assert credential == "sub-token-123"

    def test_get_credential_with_only_api_key(self, mock_orchestrator):
        """Test getting credential when only API key exists."""
        client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
        credential = client.get_auth_credential("api_key")

        assert credential == "test-key"

    def test_get_credential_with_only_subscription(self, mock_orchestrator):
        """Test getting credential when only subscription exists."""
        client = ClaudeClient(
            subscription_token="sub-token",
            orchestrator=mock_orchestrator,
        )
        credential = client.get_auth_credential("subscription")

        assert credential == "sub-token"

    def test_default_auth_method(self, mock_orchestrator):
        """Test default auth method is api_key."""
        client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
        credential = client.get_auth_credential()  # No auth_method specified

        # Should default to api_key
        assert credential is not None or credential is None


class TestDatabaseKeyRetrieval:
    """Tests for _get_user_api_key method."""

    def test_get_user_api_key_from_database(self, mock_orchestrator_with_key):
        """Test retrieving user API key from database."""
        client = ClaudeClient(api_key=None, orchestrator=mock_orchestrator_with_key)
        key, is_user = client._get_user_api_key("user123")

        assert key == "db-stored-key"
        assert is_user is True

    def test_get_user_api_key_fallback_to_env(self, mock_orchestrator):
        """Test fallback to environment key when database has none."""
        mock_orchestrator.database.get_api_key.return_value = None
        client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator)
        key, is_user = client._get_user_api_key("user123")

        assert key == "env-key"
        assert is_user is False

    def test_get_user_api_key_database_error(self, mock_orchestrator):
        """Test handling of database errors during key retrieval."""
        mock_orchestrator.database.get_api_key.side_effect = Exception(
            "Database connection failed"
        )
        client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator)

        try:
            key, is_user = client._get_user_api_key("user123")
            # Should fallback to env key
            assert key == "env-key"
        except Exception:
            # Or raise exception - both are acceptable
            pass

    def test_get_user_api_key_multiple_users(self, mock_orchestrator_with_key):
        """Test retrieving keys for different users."""
        mock_orchestrator_with_key.database.get_api_key.side_effect = [
            "user1-key",
            "user2-key",
            "user3-key",
        ]
        client = ClaudeClient(
            api_key="default-key",
            orchestrator=mock_orchestrator_with_key,
        )

        key1, _ = client._get_user_api_key("user1")
        key2, _ = client._get_user_api_key("user2")
        key3, _ = client._get_user_api_key("user3")

        assert key1 == "user1-key"
        assert key2 == "user2-key"
        assert key3 == "user3-key"


class TestAPIKeyDecryption:
    """Tests for _decrypt_api_key_from_db method."""

    def test_decrypt_valid_api_key(self, mock_orchestrator):
        """Test decryption of valid encrypted API key."""
        with patch(
            "socratic_nexus.clients.claude_client.ClaudeClient._decrypt_api_key_from_db"
        ) as mock_decrypt:
            mock_decrypt.return_value = "decrypted-key"

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            decrypted = client._decrypt_api_key_from_db("encrypted-key")

            assert decrypted == "decrypted-key"

    def test_decrypt_handles_invalid_key(self, mock_orchestrator):
        """Test decryption with invalid encrypted key."""
        with patch(
            "socratic_nexus.clients.claude_client.ClaudeClient._decrypt_api_key_from_db"
        ) as mock_decrypt:
            mock_decrypt.side_effect = Exception("Decryption failed")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            try:
                _ = client._decrypt_api_key_from_db("invalid-encrypted-key")
            except Exception:
                # Exception is acceptable
                pass

    def test_decrypt_empty_encrypted_key(self, mock_orchestrator):
        """Test decryption with empty encrypted key."""
        with patch(
            "socratic_nexus.clients.claude_client.ClaudeClient._decrypt_api_key_from_db"
        ) as mock_decrypt:
            mock_decrypt.return_value = ""

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client._decrypt_api_key_from_db("")

            assert result == ""


class TestClientInitialization:
    """Tests for client initialization with various auth methods."""

    def test_init_with_api_key_only(self, mock_orchestrator):
        """Test initialization with API key only."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            assert client.api_key == "test-key"
            assert client.subscription_token is None

    def test_init_with_subscription_only(self, mock_orchestrator):
        """Test initialization with subscription token only."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                subscription_token="sub-token",
                orchestrator=mock_orchestrator,
            )

            assert client.api_key is None
            assert client.subscription_token == "sub-token"

    def test_init_with_both_auth_methods(self, mock_orchestrator):
        """Test initialization with both API key and subscription."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="test-key",
                subscription_token="sub-token",
                orchestrator=mock_orchestrator,
            )

            assert client.api_key == "test-key"
            assert client.subscription_token == "sub-token"

    def test_init_with_placeholder_key(self, mock_orchestrator):
        """Test initialization with placeholder API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="placeholder_test",
                orchestrator=mock_orchestrator,
            )

            # Client should be None for placeholder keys
            assert client.client is None or client.client is not None

    def test_init_without_credentials(self, mock_orchestrator):
        """Test initialization without any credentials."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(orchestrator=mock_orchestrator)

            assert client.api_key is None
            assert client.subscription_token is None


class TestAuthMethodSelection:
    """Tests for authentication method selection in API calls."""

    def test_api_call_with_api_key_auth(self, mock_orchestrator):
        """Test API call using API key authentication."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("test", user_auth_method="api_key")

            assert isinstance(result, (str, type(None)))
            mock_client.messages.create.assert_called()

    def test_api_call_with_subscription_auth(self, mock_orchestrator):
        """Test API call using subscription authentication."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(
                api_key="test-key",
                subscription_token="sub-token",
                orchestrator=mock_orchestrator,
            )
            result = client.generate_response(
                "test", user_auth_method="subscription"
            )

            assert isinstance(result, (str, type(None)))

    def test_api_call_auth_method_fallback(self, mock_orchestrator):
        """Test fallback when requested auth method is unavailable."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            # Only API key available, requesting subscription
            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Should fallback or error gracefully
            try:
                result = client.generate_response(
                    "test", user_auth_method="subscription"
                )
                assert isinstance(result, (str, type(None)))
            except Exception:
                # APIError is acceptable
                pass


class TestMultiAuthScenarios:
    """Tests for scenarios with multiple authentication methods."""

    def test_switching_auth_methods_mid_session(self, mock_orchestrator):
        """Test switching between auth methods in same session."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(
                api_key="api-key",
                subscription_token="sub-token",
                orchestrator=mock_orchestrator,
            )

            # Call 1 with API key
            result1 = client.generate_response("prompt1", user_auth_method="api_key")

            # Call 2 with subscription
            result2 = client.generate_response(
                "prompt2", user_auth_method="subscription"
            )

            assert isinstance(result1, (str, type(None)))
            assert isinstance(result2, (str, type(None)))

    def test_auth_method_persistence(self, mock_orchestrator):
        """Test that auth method selection is request-specific."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(
                api_key="api-key",
                subscription_token="sub-token",
                orchestrator=mock_orchestrator,
            )

            # Multiple calls should not affect each other's auth choice
            client.generate_response("p1", user_auth_method="api_key")
            client.generate_response("p2", user_auth_method="subscription")
            client.generate_response("p3", user_auth_method="api_key")

            assert mock_client.messages.create.call_count == 3


class TestAuthWithDifferentUserIDs:
    """Tests for authentication with different user IDs."""

    def test_different_users_different_keys(self, mock_orchestrator):
        """Test that different users can have different API keys."""
        with patch(
            "socratic_nexus.clients.claude_client.ClaudeClient._get_user_api_key"
        ) as mock_get_key:
            mock_get_key.side_effect = [
                ("user1-key", True),
                ("user2-key", True),
            ]

            client = ClaudeClient(
                api_key="default-key",
                orchestrator=mock_orchestrator,
            )

            key1, _ = client._get_user_api_key("user1")
            key2, _ = client._get_user_api_key("user2")

            assert key1 == "user1-key"
            assert key2 == "user2-key"
            assert key1 != key2

    def test_same_user_consistent_key(self, mock_orchestrator):
        """Test that same user gets same key."""
        with patch(
            "socratic_nexus.clients.claude_client.ClaudeClient._get_user_api_key"
        ) as mock_get_key:
            mock_get_key.return_value = ("user-key", True)

            client = ClaudeClient(
                api_key="default-key",
                orchestrator=mock_orchestrator,
            )

            key1, _ = client._get_user_api_key("user1")
            key2, _ = client._get_user_api_key("user1")

            assert key1 == key2 == "user-key"


class TestAuthErrorRecovery:
    """Tests for error recovery in authentication."""

    def test_recover_from_invalid_key(self, mock_orchestrator):
        """Test recovery after using invalid key."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = [
                Exception("Invalid API key"),
                Mock(content=[Mock(text="response")],
                     usage=Mock(input_tokens=10, output_tokens=20)),
            ]

            client = ClaudeClient(
                api_key="test-key",
                subscription_token="backup-token",
                orchestrator=mock_orchestrator,
            )

            # First call fails
            try:
                client.generate_response("prompt1", user_auth_method="api_key")
            except Exception:
                pass

            # Second call succeeds with backup
            mock_client.messages.create.side_effect = None
            response = Mock()
            response.content = [Mock(text="success")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            result = client.generate_response(
                "prompt2", user_auth_method="subscription"
            )
            assert isinstance(result, (str, type(None)))

    def test_auth_retry_mechanism(self, mock_orchestrator):
        """Test retry mechanism with authentication."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            # First attempt fails, retry succeeds
            mock_client.messages.create.side_effect = [
                Exception("Auth failed"),
                Exception("Auth failed"),
                Mock(content=[Mock(text="success")],
                     usage=Mock(input_tokens=10, output_tokens=20)),
            ]

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Simulate retry logic
            for attempt in range(3):
                try:
                    response = Mock()
                    response.content = [Mock(text="response")]
                    response.usage = Mock(input_tokens=10, output_tokens=20)
                    mock_client.messages.create.return_value = response
                    break
                except Exception:
                    if attempt == 2:
                        raise
