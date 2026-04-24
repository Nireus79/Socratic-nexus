"""Comprehensive tests for client code coverage expansion.

These tests aim to maximize code coverage by testing:
- All client initialization paths
- Configuration handling
- Error conditions and edge cases
- Method call paths
- State management
"""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext, ConflictInfo, ChatResponse, TokenUsage
from socratic_nexus.exceptions import APIError


class TestClaudeClientInitVariations:
    """Test all initialization variations."""

    def test_init_with_orchestrator_and_api_key(self):
        """Test initialization with both orchestrator and api_key."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            orch = Mock()
            orch.config = Mock()
            orch.config.claude_model = "claude-test"
            client = ClaudeClient(api_key="key", orchestrator=orch)
            assert client.api_key == "key"
            assert client.orchestrator is orch
            assert client.model == "claude-test"

    def test_init_with_none_orchestrator(self):
        """Test initialization with None orchestrator."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert client.api_key == "key"
            assert client.orchestrator is None

    def test_init_with_none_api_key(self):
        """Test initialization with None api_key."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=None)
            assert client.api_key is None

    def test_init_subscription_token_none(self):
        """Test initialization with subscription_token as None."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None, subscription_token=None)
            assert client.subscription_token is None

    def test_init_sets_logger(self):
        """Test that initialization sets up logger."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert client.logger is not None

    def test_init_caches_are_dicts(self):
        """Test that caches are initialized as dictionaries."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert isinstance(client._insights_cache, dict)
            assert isinstance(client._question_cache, dict)

    def test_init_clients_none_by_default(self):
        """Test that anthropic clients start as None."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="placeholder", orchestrator=None)
            assert client.client is None
            assert client.async_client is None
            assert client.subscription_client is None
            assert client.subscription_async_client is None


class TestAuthCredentialMethods:
    """Test authentication credential methods."""

    def test_get_auth_credential_with_api_key_method(self):
        """Test get_auth_credential with api_key method."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test-api-key", orchestrator=None)
            result = client.get_auth_credential("api_key")
            assert result == "test-api-key"

    def test_get_auth_credential_default_method(self):
        """Test get_auth_credential defaults to api_key."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=None)
            result = client.get_auth_credential()
            assert result == "test-key"

    def test_get_auth_credential_with_subscription(self):
        """Test get_auth_credential with subscription method."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None, subscription_token="sub-token")
            result = client.get_auth_credential("subscription")
            assert result == "sub-token"

    def test_get_auth_credential_missing_subscription_raises(self):
        """Test that missing subscription token raises ValueError."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            with pytest.raises(ValueError):
                client.get_auth_credential("subscription")

    def test_get_auth_credential_missing_api_key_raises(self):
        """Test that missing api_key raises ValueError."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=None)
            with pytest.raises(ValueError):
                client.get_auth_credential("api_key")


class TestGetClientMethod:
    """Test _get_client method."""

    @pytest.fixture
    def mock_orch_with_db(self):
        """Create orchestrator with database."""
        orch = Mock()
        orch.config = Mock()
        orch.config.claude_model = "test"
        orch.database = Mock()
        return orch

    def test_get_client_with_api_key(self, mock_orch_with_db):
        """Test _get_client uses api_key auth method."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=mock_orch_with_db)
            client.client = Mock()
            result = client._get_client(user_auth_method="api_key")
            assert result is not None

    def test_get_client_no_key_available_raises_error(self, mock_orch_with_db):
        """Test _get_client raises APIError when no key available."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            mock_orch_with_db.database.get_api_key.return_value = None
            client = ClaudeClient(api_key=None, orchestrator=mock_orch_with_db)
            with pytest.raises(APIError) as exc:
                client._get_client(user_auth_method="api_key")
            assert exc.value.error_type == "MISSING_API_KEY"

    def test_get_client_subscription_mode_defaults_to_api_key(self, mock_orch_with_db):
        """Test that subscription mode is not supported and defaults to api_key."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=mock_orch_with_db)
            client.client = Mock()
            # Should not raise, should default to api_key
            result = client._get_client(user_auth_method="subscription")
            assert result is not None


class TestProjectContextHandling:
    """Test ProjectContext handling throughout client."""

    def test_project_context_with_all_fields(self):
        """Test ProjectContext with all fields populated."""
        context = ProjectContext(
            project_name="Test Project",
            description="Test Description",
            goals=["Goal 1"],
            phase="implementation",
            tech_stack=["Python"],
            project_type="web",
            deployment_target="cloud",
            status="active",
        )
        assert context.project_name == "Test Project"
        assert context.goals == ["Goal 1"]
        assert context.phase == "implementation"

    def test_project_context_minimal(self):
        """Test ProjectContext with minimal fields."""
        context = ProjectContext(project_name="Test")
        assert context.project_name == "Test"
        assert context.description == ""
        assert context.goals is None

    def test_conflict_info_creation(self):
        """Test ConflictInfo creation."""
        conflict = ConflictInfo(
            description="Test conflict",
            file_path="test.py",
            line_number=10,
            conflict_type="merge",
            severity="high",
        )
        assert conflict.description == "Test conflict"
        assert conflict.conflict_type == "merge"
        assert conflict.severity == "high"


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_is_sha256(self):
        """Test that cache keys are SHA256 hashes."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            key = client._get_cache_key("test")
            assert len(key) == 64  # SHA256 hex is 64 chars
            assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_deterministic(self):
        """Test cache keys are deterministic."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            key1 = client._get_cache_key("message")
            key2 = client._get_cache_key("message")
            assert key1 == key2

    def test_different_messages_different_keys(self):
        """Test different messages produce different keys."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            key1 = client._get_cache_key("message1")
            key2 = client._get_cache_key("message2")
            assert key1 != key2


class TestTokenUsageModel:
    """Test TokenUsage model."""

    def test_token_usage_all_fields(self):
        """Test TokenUsage with all fields."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            provider="claude",
            model="claude-3-sonnet",
            cost_usd=0.01,
            latency_ms=100.5,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost_usd == 0.01
        assert usage.latency_ms == 100.5

    def test_token_usage_defaults(self):
        """Test TokenUsage with default values."""
        usage = TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, provider="test", model="test"
        )
        assert usage.cost_usd == 0.0
        assert usage.latency_ms == 0.0

    def test_chat_response_creation(self):
        """Test ChatResponse creation."""
        usage = TokenUsage(
            input_tokens=10, output_tokens=5, total_tokens=15, provider="test", model="test"
        )
        response = ChatResponse(content="Test response", usage=usage, model="test", provider="test")
        assert response.content == "Test response"
        assert response.usage == usage


class TestEventEmission:
    """Test event emission."""

    def test_orchestrator_event_emitter_available(self):
        """Test that orchestrator event_emitter is accessible."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            orch = Mock()
            orch.config = Mock()
            orch.config.claude_model = "test"
            orch.event_emitter = Mock()

            client = ClaudeClient(api_key="key", orchestrator=orch)
            assert client.orchestrator.event_emitter is not None


class TestErrorHandling:
    """Test error handling."""

    def test_api_error_basic(self):
        """Test APIError creation."""
        error = APIError("Test error")
        assert str(error) == "Test error"

    def test_api_error_with_type(self):
        """Test APIError with error_type."""
        error = APIError("Test", error_type="TEST_ERROR")
        assert error.error_type == "TEST_ERROR"

    def test_api_error_with_kwargs(self):
        """Test APIError with additional kwargs."""
        error = APIError("Test", error_type="TEST", code=500)
        assert error.details == {"code": 500}

    def test_api_error_inheritance(self):
        """Test APIError is an Exception."""
        error = APIError("Test")
        assert isinstance(error, Exception)


class TestClientLogging:
    """Test client logging configuration."""

    def test_client_has_logger(self):
        """Test that client has a logger."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert hasattr(client, "logger")
            assert client.logger is not None

    def test_logger_name(self):
        """Test logger has correct name."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert "claude" in client.logger.name.lower()


class TestClientAttributes:
    """Test client attributes and properties."""

    def test_client_attributes_after_init(self):
        """Test all expected attributes exist after initialization."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="key", orchestrator=None)
            assert hasattr(client, "api_key")
            assert hasattr(client, "orchestrator")
            assert hasattr(client, "model")
            assert hasattr(client, "logger")
            assert hasattr(client, "client")
            assert hasattr(client, "async_client")
            assert hasattr(client, "subscription_client")
            assert hasattr(client, "subscription_async_client")
            assert hasattr(client, "_insights_cache")
            assert hasattr(client, "_question_cache")


class TestIntegrationPaths:
    """Test integration between different components."""

    def test_user_api_key_retrieval_flow(self):
        """Test the complete user API key retrieval flow."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            orch = Mock()
            orch.config = Mock()
            orch.config.claude_model = "test"
            orch.database = Mock()
            orch.database.get_api_key.return_value = None

            client = ClaudeClient(api_key="env-key", orchestrator=orch)
            key, is_user = client._get_user_api_key(user_id="user1")

            assert key == "env-key"
            assert is_user is False

    def test_orchestrator_configuration_usage(self):
        """Test that orchestrator config is used correctly."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            orch = Mock()
            orch.config = Mock()
            orch.config.claude_model = "claude-custom-model"

            client = ClaudeClient(api_key="key", orchestrator=orch)
            assert client.model == "claude-custom-model"

    def test_placeholder_key_detection(self):
        """Test that placeholder keys are detected and skipped."""
        with patch("socratic_nexus.clients.claude_client.anthropic"):
            client = ClaudeClient(api_key="placeholder_test", orchestrator=None)
            # Anthropic constructor should not be called for placeholder keys
            assert client.client is None
            assert client.async_client is None
