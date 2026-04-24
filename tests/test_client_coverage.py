"""Comprehensive tests for client methods to increase code coverage.

Focus on testable methods that don't require actual API calls:
- Cache key generation and management
- Authentication credential handling
- Error conditions and edge cases
- Client initialization variations
"""

import hashlib
import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.exceptions import APIError
from socratic_nexus.models import TokenUsage


class TestClaudeClientCacheKeys:
    """Test cache key generation and management."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = Mock()
        orchestrator.config = Mock()
        orchestrator.config.claude_model = "claude-3-sonnet-20240229"
        return orchestrator

    @pytest.fixture
    def claude_client(self, mock_orchestrator):
        """Create Claude client with mocked anthropic."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
        return client

    def test_cache_key_deterministic(self, claude_client):
        """Test that cache keys are deterministic."""
        message = "Test message for caching"
        key1 = claude_client._get_cache_key(message)
        key2 = claude_client._get_cache_key(message)
        assert key1 == key2

    def test_cache_key_different_messages(self, claude_client):
        """Test that different messages produce different cache keys."""
        key1 = claude_client._get_cache_key("Message 1")
        key2 = claude_client._get_cache_key("Message 2")
        assert key1 != key2

    def test_cache_key_format(self, claude_client):
        """Test that cache key is SHA256 hex string."""
        message = "Test"
        key = claude_client._get_cache_key(message)
        expected = hashlib.sha256(message.encode()).hexdigest()
        assert key == expected
        assert len(key) == 64  # SHA256 hex is 64 chars


class TestClaudeClientCredentials:
    """Test credential retrieval and handling."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = Mock()
        orchestrator.config = Mock()
        orchestrator.config.claude_model = "claude-3-sonnet-20240229"
        return orchestrator

    @pytest.fixture
    def claude_client_no_key(self, mock_orchestrator):
        """Create Claude client without API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=mock_orchestrator)
        return client

    @pytest.fixture
    def claude_client_with_subscription(self, mock_orchestrator):
        """Create Claude client with subscription token."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(
                api_key="test-key",
                orchestrator=mock_orchestrator,
                subscription_token="sub-token",
            )
        return client

    def test_get_auth_credential_defaults_to_api_key(self, mock_orchestrator):
        """Test that get_auth_credential defaults to api_key method."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="my-api-key", orchestrator=mock_orchestrator)
        credential = client.get_auth_credential()
        assert credential == "my-api-key"

    def test_get_auth_credential_subscription(self, claude_client_with_subscription):
        """Test subscription credential retrieval."""
        credential = claude_client_with_subscription.get_auth_credential("subscription")
        assert credential == "sub-token"

    def test_get_auth_credential_missing_subscription(self, claude_client_no_key):
        """Test error when subscription requested but not configured."""
        with pytest.raises(ValueError, match="Subscription token"):
            claude_client_no_key.get_auth_credential("subscription")

    def test_get_auth_credential_missing_api_key(self, claude_client_no_key):
        """Test error when api_key requested but not configured."""
        with pytest.raises(ValueError, match="API key"):
            claude_client_no_key.get_auth_credential("api_key")


class TestClaudeClientInitialization:
    """Test various initialization scenarios."""

    def test_client_init_with_placeholder_key(self):
        """Test that placeholder keys don't initialize actual clients."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="placeholder_key", orchestrator=None)
            assert client.api_key == "placeholder_key"
            assert client.client is None  # Should not initialize client

    def test_client_init_without_orchestrator_uses_default_model(self):
        """Test that client without orchestrator uses default model."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=None)
            assert client.model == "claude-haiku-4-5-20251001"

    def test_client_caches_initialized(self):
        """Test that client initializes empty caches."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=None)
            assert client._insights_cache == {}
            assert client._question_cache == {}

    def test_client_with_valid_api_key_initializes_clients(self):
        """Test that valid API key initializes anthropic clients."""
        with patch("socratic_nexus.clients.claude_client.anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_async_client = Mock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_anthropic.AsyncAnthropic.return_value = mock_async_client

            ClaudeClient(api_key="sk-real-key", orchestrator=None)

            # Should have called Anthropic constructors
            mock_anthropic.Anthropic.assert_called_with(api_key="sk-real-key")
            mock_anthropic.AsyncAnthropic.assert_called_with(api_key="sk-real-key")


class TestClaudeClientGetUserApiKey:
    """Test user-specific API key retrieval."""

    @pytest.fixture
    def mock_orchestrator_with_db(self):
        """Create mock orchestrator with database."""
        orchestrator = Mock()
        orchestrator.config = Mock()
        orchestrator.config.claude_model = "claude-3-sonnet-20240229"
        orchestrator.database = Mock()
        return orchestrator

    def test_get_user_api_key_no_user_id(self, mock_orchestrator_with_db):
        """Test that missing user_id returns environment API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator_with_db)
            key, is_user_specific = client._get_user_api_key(user_id=None)
            assert key == "env-key"
            assert is_user_specific is False

    def test_get_user_api_key_missing_from_db(self, mock_orchestrator_with_db):
        """Test fallback to env key when user key not in database."""
        mock_orchestrator_with_db.database.get_api_key.return_value = None
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator_with_db)
            key, is_user_specific = client._get_user_api_key(user_id="user123")
            assert key == "env-key"
            assert is_user_specific is False

    def test_get_user_api_key_db_error_fallback(self, mock_orchestrator_with_db):
        """Test fallback to env key when database raises error."""
        mock_orchestrator_with_db.database.get_api_key.side_effect = Exception("DB error")
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator_with_db)
            key, is_user_specific = client._get_user_api_key(user_id="user123")
            assert key == "env-key"
            assert is_user_specific is False


class TestTokenUsageModeling:
    """Test token usage data model."""

    def test_token_usage_creation(self):
        """Test creating TokenUsage instance."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            provider="claude",
            model="claude-3-sonnet",
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.provider == "claude"
        assert usage.model == "claude-3-sonnet"

    def test_token_usage_with_cost(self):
        """Test TokenUsage with cost calculation."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            provider="claude",
            model="claude-3-sonnet",
            cost_usd=0.015,
        )
        assert usage.cost_usd == 0.015

    def test_token_usage_cost_default(self):
        """Test TokenUsage with default cost value."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            provider="claude",
            model="claude-3-sonnet",
        )
        assert usage.cost_usd == 0.0


class TestAPIErrorHandling:
    """Test API error handling and types."""

    def test_api_error_with_missing_api_key(self):
        """Test APIError for missing API key scenario."""
        error = APIError("Missing API key", error_type="MISSING_API_KEY", status_code=None)
        assert error.error_type == "MISSING_API_KEY"
        assert "Missing API key" in str(error)

    def test_api_error_with_details(self):
        """Test APIError with additional details."""
        error = APIError("Rate limit exceeded", error_type="RATE_LIMIT", retry_after=60)
        assert error.details == {"retry_after": 60}
        assert error.error_type == "RATE_LIMIT"

    def test_api_error_inheritance(self):
        """Test that APIError inherits from Exception."""
        error = APIError("Test error", error_type="TEST")
        assert isinstance(error, Exception)


class TestClientMethodPresence:
    """Test that all expected methods exist on clients."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = Mock()
        orchestrator.config = Mock()
        orchestrator.config.claude_model = "claude-3-sonnet-20240229"
        return orchestrator

    def test_claude_client_has_required_methods(self, mock_orchestrator):
        """Test that ClaudeClient has all required public methods."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)

        required_methods = [
            "get_auth_credential",
            "_get_cache_key",
            "_get_user_api_key",
            "_decrypt_api_key_from_db",
        ]

        for method_name in required_methods:
            assert hasattr(client, method_name), f"Missing method: {method_name}"
            assert callable(getattr(client, method_name))

    def test_claude_client_attributes_exist(self, mock_orchestrator):
        """Test that ClaudeClient has expected attributes."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)

        expected_attrs = ["api_key", "orchestrator", "model", "_insights_cache"]
        for attr_name in expected_attrs:
            assert hasattr(client, attr_name), f"Missing attribute: {attr_name}"
