"""Extended tests targeting specific code paths for coverage increase.

Tests focusing on:
- Method signature validation
- Parameter handling
- Error paths
- Initialization logic
"""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext
from socratic_nexus.exceptions import APIError


class TestClientInitializationDetails:
    """Test detailed initialization logic."""

    def test_real_api_key_initializes_sync_client(self):
        """Test that a real API key initializes sync anthropic client."""
        with patch("socratic_nexus.clients.claude_client.anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.Anthropic.return_value = mock_client

            client = ClaudeClient(api_key="sk-real-key", orchestrator=None)

            mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-real-key")
            assert client.client == mock_client

    def test_real_api_key_initializes_async_client(self):
        """Test that a real API key initializes async anthropic client."""
        with patch("socratic_nexus.clients.claude_client.anthropic") as mock_anthropic:
            mock_async_client = Mock()
            mock_anthropic.AsyncAnthropic.return_value = mock_async_client

            client = ClaudeClient(api_key="sk-real-key", orchestrator=None)

            mock_anthropic.AsyncAnthropic.assert_called_once_with(api_key="sk-real-key")
            assert client.async_client == mock_async_client

    def test_subscription_token_initializes_subscription_client(self):
        """Test that subscription token initializes subscription client."""
        with patch("socratic_nexus.clients.claude_client.anthropic") as mock_anthropic:
            mock_sub_client = Mock()
            mock_anthropic.Anthropic.return_value = mock_sub_client

            client = ClaudeClient(api_key="key", orchestrator=None, subscription_token="sub-token")

            # Should be called with subscription_token
            assert mock_anthropic.Anthropic.call_count >= 1
            assert client.subscription_client == mock_sub_client

    def test_subscription_token_initializes_async_client(self):
        """Test that subscription token initializes async subscription client."""
        with patch("socratic_nexus.clients.claude_client.anthropic") as mock_anthropic:
            mock_async_sub = Mock()
            mock_anthropic.AsyncAnthropic.return_value = mock_async_sub

            client = ClaudeClient(api_key="key", orchestrator=None, subscription_token="sub-token")

            assert client.subscription_async_client == mock_async_sub


class TestModelSelectionFromOrchestrator:
    """Test model selection from orchestrator config."""

    def test_uses_orchestrator_model_config(self):
        """Test that client uses model from orchestrator config."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            orch = Mock()
            orch.config = Mock()
            orch.config.claude_model = "claude-custom-model-123"

            client = ClaudeClient(api_key="key", orchestrator=orch)
            assert client.model == "claude-custom-model-123"

    def test_uses_default_model_without_orchestrator(self):
        """Test that client uses default model without orchestrator."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert client.model == "claude-haiku-4-5-20251001"

    def test_default_model_is_string(self):
        """Test that model is a string value."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert isinstance(client.model, str)
            assert len(client.model) > 0


class TestGetUserApiKeyVariations:
    """Test _get_user_api_key method variations."""

    @pytest.fixture
    def mock_orch(self):
        """Create mock orchestrator."""
        orch = Mock()
        orch.config = Mock()
        orch.config.claude_model = "test"
        orch.database = Mock()
        return orch

    def test_returns_tuple_with_boolean(self, mock_orch):
        """Test that _get_user_api_key returns (key, bool) tuple."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="env-key", orchestrator=mock_orch)
            mock_orch.database.get_api_key.return_value = None

            result = client._get_user_api_key(user_id="user1")

            assert isinstance(result, tuple)
            assert len(result) == 2
            key, is_user_specific = result
            assert isinstance(is_user_specific, bool)

    def test_fallback_when_database_returns_none(self, mock_orch):
        """Test fallback to environment key when DB returns None."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="env-api-key", orchestrator=mock_orch)
            mock_orch.database.get_api_key.return_value = None

            key, is_user = client._get_user_api_key(user_id="user1")

            assert key == "env-api-key"
            assert is_user is False

    def test_database_exception_handled(self, mock_orch):
        """Test that database exceptions are handled gracefully."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="env-key", orchestrator=mock_orch)
            mock_orch.database.get_api_key.side_effect = RuntimeError("DB error")

            key, is_user = client._get_user_api_key(user_id="user1")

            assert key == "env-key"
            assert is_user is False

    def test_no_user_id_returns_env_key(self, mock_orch):
        """Test that None user_id returns environment key."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="env-key", orchestrator=mock_orch)

            key, is_user = client._get_user_api_key(user_id=None)

            assert key == "env-key"
            assert is_user is False


class TestCacheOperations:
    """Test cache operations."""

    def test_insights_cache_starts_empty(self):
        """Test that insights cache starts empty."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert len(client._insights_cache) == 0

    def test_question_cache_starts_empty(self):
        """Test that question cache starts empty."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert len(client._question_cache) == 0

    def test_cache_key_function_returns_string(self):
        """Test that _get_cache_key returns a string."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            result = client._get_cache_key("test message")
            assert isinstance(result, str)

    def test_cache_key_hex_format(self):
        """Test that cache key is in hexadecimal format."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            result = client._get_cache_key("test")
            # Should only contain hex characters
            assert all(c in "0123456789abcdef" for c in result)


class TestAuthenticationMethods:
    """Test authentication method handling."""

    def test_get_auth_credential_unknown_method_uses_api_key(self):
        """Test that unknown auth method defaults to api_key."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test-api-key", orchestrator=None)
            # Pass unknown method
            result = client.get_auth_credential("unknown_method")
            assert result == "test-api-key"

    def test_get_auth_credential_empty_string_method(self):
        """Test with empty string as method."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=None)
            # Empty string should default to api_key
            result = client.get_auth_credential("")
            assert result == "test-key"

    def test_get_auth_credential_case_sensitive(self):
        """Test that method names are case sensitive."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=None)
            # Uppercase should not match 'subscription'
            result = client.get_auth_credential("SUBSCRIPTION")
            assert result == "test-key"  # Should default to api_key


class TestClientStateMaintenance:
    """Test client state maintenance."""

    def test_api_key_stored_correctly(self):
        """Test that API key is stored correctly."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            test_key = "sk-test-12345"
            client = ClaudeClient(api_key=test_key, orchestrator=None)
            assert client.api_key == test_key

    def test_orchestrator_reference_maintained(self):
        """Test that orchestrator reference is maintained."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            orch = Mock()
            orch.config = Mock()
            orch.config.claude_model = "test"
            client = ClaudeClient(api_key="key", orchestrator=orch)
            assert client.orchestrator is orch

    def test_subscription_token_stored_correctly(self):
        """Test that subscription token is stored correctly."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            sub_token = "sub-12345"
            client = ClaudeClient(api_key="key", orchestrator=None, subscription_token=sub_token)
            assert client.subscription_token == sub_token


class TestProjectContextIntegration:
    """Test ProjectContext integration."""

    def test_project_context_fields_accessible(self):
        """Test that all ProjectContext fields are accessible."""
        context = ProjectContext(
            project_name="TestProj",
            description="Test Description",
            phase="testing",
            goals=["Goal1"],
        )
        assert context.project_name == "TestProj"
        assert context.description == "Test Description"
        assert context.phase == "testing"
        assert context.goals == ["Goal1"]

    def test_project_context_optional_fields(self):
        """Test that optional fields can be None."""
        context = ProjectContext(project_name="Test")
        assert context.goals is None
        assert context.phase is None
        assert context.tech_stack is None

    def test_project_context_with_metadata(self):
        """Test ProjectContext with metadata."""
        metadata = {"key": "value"}
        context = ProjectContext(project_name="Test", metadata=metadata)
        assert context.metadata == metadata


class TestErrorTypeHandling:
    """Test error type handling."""

    def test_api_error_missing_api_key_type(self):
        """Test APIError with MISSING_API_KEY type."""
        error = APIError("No key", error_type="MISSING_API_KEY")
        assert error.error_type == "MISSING_API_KEY"
        assert "No key" in str(error)

    def test_api_error_custom_type(self):
        """Test APIError with custom error type."""
        error = APIError("Error", error_type="CUSTOM_ERROR")
        assert error.error_type == "CUSTOM_ERROR"

    def test_api_error_default_type(self):
        """Test APIError with default type."""
        error = APIError("Error")
        assert error.error_type == "unknown"


class TestPlaceholderKeyHandling:
    """Test placeholder key handling."""

    def test_placeholder_key_prefix_detected(self):
        """Test that placeholder_ prefix is detected."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="placeholder_test", orchestrator=None)
            assert client.client is None

    def test_placeholder_does_not_initialize_clients(self):
        """Test that placeholder keys don't initialize anthropic clients."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="placeholder_anything", orchestrator=None)
            assert client.client is None
            assert client.async_client is None
            assert client.subscription_client is None
            assert client.subscription_async_client is None

    def test_sk_prefix_is_real_key(self):
        """Test that sk- prefix is treated as real key."""
        with patch("socratic_nexus.clients.claude_client.anthropic") as mock_anth:
            mock_anth.Anthropic.return_value = Mock()
            mock_anth.AsyncAnthropic.return_value = Mock()

            client = ClaudeClient(api_key="sk-real-key", orchestrator=None)

            # Should have initialized clients
            assert client.client is not None
            assert client.async_client is not None
