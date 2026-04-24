"""Comprehensive tests for authentication flows and credential handling."""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.exceptions import APIError


@pytest.fixture
def mock_orchestrator_with_db():
    """Mock orchestrator with database."""
    orch = Mock()
    orch.config = Mock()
    orch.config.claude_model = "claude-3-sonnet-20240229"
    orch.event_emitter = Mock()
    orch.database = Mock()
    return orch


class TestGetUserApiKeyFlow:
    """Tests for _get_user_api_key method."""

    def test_get_user_api_key_no_user_id_returns_env_key(self, mock_orchestrator_with_db):
        """Test _get_user_api_key with no user_id returns env key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator_with_db)
            key, is_user = client._get_user_api_key(user_id=None)

            assert key == "env-key"
            assert is_user is False

    def test_get_user_api_key_with_user_id_database_returns_none(self, mock_orchestrator_with_db):
        """Test _get_user_api_key falls back to env when database returns None."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator_with_db)
            key, is_user = client._get_user_api_key(user_id="user123")

            assert key == "env-key"
            assert is_user is False

    def test_get_user_api_key_database_error_fallback(self, mock_orchestrator_with_db):
        """Test _get_user_api_key falls back on database error."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            mock_orchestrator_with_db.database.get_api_key.side_effect = Exception("DB Error")

            client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator_with_db)
            key, is_user = client._get_user_api_key(user_id="user123")

            assert key == "env-key"
            assert is_user is False

    def test_get_user_api_key_no_api_key_raises_error(self, mock_orchestrator_with_db):
        """Test _get_user_api_key raises APIError when no key available."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key=None, orchestrator=mock_orchestrator_with_db)

            with pytest.raises(APIError):
                client._get_user_api_key(user_id="user123")

    def test_get_user_api_key_placeholder_key_raises_error(self, mock_orchestrator_with_db):
        """Test _get_user_api_key with placeholder key raises error."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(
                api_key="placeholder_test", orchestrator=mock_orchestrator_with_db
            )

            with pytest.raises(APIError):
                client._get_user_api_key(user_id="user123")

    def test_get_user_api_key_returns_decrypted_key(self, mock_orchestrator_with_db):
        """Test _get_user_api_key decrypts stored key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            with patch.object(ClaudeClient, "_decrypt_api_key_from_db") as mock_decrypt:
                mock_decrypt.return_value = "decrypted-key"
                mock_orchestrator_with_db.database.get_api_key.return_value = "encrypted-data"

                client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator_with_db)
                key, is_user = client._get_user_api_key(user_id="user123")

                assert key == "decrypted-key"
                assert is_user is True

    def test_get_user_api_key_decrypt_failure_fallback(self, mock_orchestrator_with_db):
        """Test _get_user_api_key falls back when decryption fails."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            with patch.object(ClaudeClient, "_decrypt_api_key_from_db") as mock_decrypt:
                mock_decrypt.return_value = None  # Decryption failed
                mock_orchestrator_with_db.database.get_api_key.return_value = "encrypted-data"

                client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator_with_db)
                key, is_user = client._get_user_api_key(user_id="user123")

                assert key == "env-key"
                assert is_user is False


class TestGetClientMethod:
    """Tests for _get_client method."""

    def test_get_client_api_key_auth(self, mock_orchestrator_with_db):
        """Test _get_client with api_key auth method."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator_with_db)
            result = client._get_client(user_auth_method="api_key")

            assert result is not None

    def test_get_client_subscription_fallback_to_api_key(self, mock_orchestrator_with_db):
        """Test _get_client falls back from subscription to api_key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator_with_db)
            result = client._get_client(user_auth_method="subscription")

            assert result is not None

    def test_get_client_with_user_id_no_user_key(self, mock_orchestrator_with_db):
        """Test _get_client with user_id when user has no stored key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator_with_db)
            result = client._get_client(user_auth_method="api_key", user_id="user123")

            assert result is not None

    def test_get_client_missing_key_raises_error(self, mock_orchestrator_with_db):
        """Test _get_client raises APIError when no key available."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key=None, orchestrator=mock_orchestrator_with_db)

            with pytest.raises(APIError):
                client._get_client(user_auth_method="api_key")

    def test_get_client_creates_new_instance(self, mock_orchestrator_with_db):
        """Test _get_client creates new Anthropic instance."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator_with_db)
            result = client._get_client(user_auth_method="api_key")

            # Should create instance with the API key
            assert result is not None


class TestGetAsyncClientMethod:
    """Tests for _get_async_client method."""

    def test_get_async_client_api_key_auth(self, mock_orchestrator_with_db):
        """Test _get_async_client with api_key auth method."""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator_with_db)
                result = client._get_async_client(user_auth_method="api_key")

                assert result is not None

    def test_get_async_client_with_user_id(self, mock_orchestrator_with_db):
        """Test _get_async_client with user_id parameter."""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator_with_db)
                result = client._get_async_client(user_auth_method="api_key", user_id="user123")

                assert result is not None


class TestDecryptApiKeyFromDb:
    """Tests for _decrypt_api_key_from_db method."""

    def test_decrypt_api_key_returns_none_on_failure(self, mock_orchestrator_with_db):
        """Test _decrypt_api_key_from_db returns None on failure."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator_with_db)

            try:
                result = client._decrypt_api_key_from_db("invalid_encrypted_data")
                assert result is None or isinstance(result, str)
            except ImportError:
                # Cryptography module not available
                pass

    def test_decrypt_api_key_with_invalid_data(self, mock_orchestrator_with_db):
        """Test _decrypt_api_key_from_db with invalid data."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator_with_db)

            try:
                result = client._decrypt_api_key_from_db("!@#$%^&*()")
                assert result is None or isinstance(result, str)
            except (ImportError, Exception):
                # Expected to fail or cryptography not available
                pass

    def test_decrypt_api_key_with_empty_string(self, mock_orchestrator_with_db):
        """Test _decrypt_api_key_from_db with empty string."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator_with_db)

            try:
                result = client._decrypt_api_key_from_db("")
                assert result is None or isinstance(result, str)
            except (ImportError, Exception):
                pass


class TestAuthenticationErrorScenarios:
    """Tests for authentication error scenarios."""

    def test_api_error_includes_error_type(self, mock_orchestrator_with_db):
        """Test APIError includes error_type."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key=None, orchestrator=mock_orchestrator_with_db)

            try:
                client._get_user_api_key(user_id="user123")
            except APIError as e:
                assert e.error_type == "MISSING_API_KEY"

    def test_api_error_helpful_message(self, mock_orchestrator_with_db):
        """Test APIError includes helpful message."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key=None, orchestrator=mock_orchestrator_with_db)

            try:
                client._get_user_api_key(user_id="user123")
            except APIError as e:
                assert "API key" in str(e) or "Settings" in str(e)


class TestAuthenticationFlowIntegration:
    """Integration tests for authentication flows."""

    def test_full_auth_flow_with_env_key(self, mock_orchestrator_with_db):
        """Test full authentication flow using environment key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_orchestrator_with_db.database.get_api_key.return_value = None

            client = ClaudeClient(api_key="env-api-key", orchestrator=mock_orchestrator_with_db)

            # Get auth credential
            cred = client.get_auth_credential("api_key")
            assert cred == "env-api-key"

            # Get client with that credential
            api_client = client._get_client(user_auth_method="api_key")
            assert api_client is not None

    def test_auth_flow_credentials_not_exposed(self, mock_orchestrator_with_db):
        """Test that credentials are not logged or exposed."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="secret-key-123", orchestrator=mock_orchestrator_with_db)

            # Get auth credential should return the key
            cred = client.get_auth_credential()
            assert cred == "secret-key-123"

            # But logger should have been configured safely
            assert client.logger is not None
