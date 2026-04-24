"""Tests for Claude API client."""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext, ConflictInfo, TokenUsage


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator."""
    orchestrator = Mock()
    orchestrator.config = Mock()
    orchestrator.config.claude_model = "claude-3-sonnet-20240229"
    orchestrator.event_emitter = Mock()
    orchestrator.context_analyzer = Mock()
    return orchestrator


@pytest.fixture
def claude_client(mock_orchestrator):
    """Create a ClaudeClient instance with mocked orchestrator."""
    with patch("socratic_nexus.clients.claude_client.anthropic"):
        client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
    return client


def test_claude_client_initialization(mock_orchestrator):
    """Test ClaudeClient initialization."""
    with patch("socratic_nexus.clients.claude_client.anthropic"):
        client = ClaudeClient(api_key="test-api-key", orchestrator=mock_orchestrator)
        assert client.api_key == "test-api-key"
        assert client.orchestrator == mock_orchestrator


def test_claude_client_without_orchestrator():
    """Test ClaudeClient initialization without orchestrator."""
    with patch("socratic_nexus.clients.claude_client.anthropic"):
        client = ClaudeClient(api_key="test-api-key", orchestrator=None)
        assert client.api_key == "test-api-key"
        assert client.orchestrator is None


def test_claude_client_get_auth_credential_api_key(claude_client):
    """Test getting auth credential with api_key method."""
    credential = claude_client.get_auth_credential("api_key")
    assert credential == "test-key"


def test_claude_client_get_auth_credential_subscription(claude_client):
    """Test getting auth credential with subscription method."""
    claude_client.subscription_token = "sub-token"
    credential = claude_client.get_auth_credential("subscription")
    assert credential == "sub-token"


def test_claude_client_get_auth_credential_missing_subscription(claude_client):
    """Test getting auth credential with missing subscription token."""
    claude_client.subscription_token = None
    with pytest.raises(ValueError, match="Subscription token"):
        claude_client.get_auth_credential("subscription")


def test_claude_client_get_auth_credential_unknown_method(claude_client):
    """Test getting auth credential with unknown method defaults to api_key."""
    # Unknown methods are treated as api_key in claude_client
    credential = claude_client.get_auth_credential("unknown")
    assert credential == "test-key"


def test_decrypt_api_key_invalid(claude_client):
    """Test decryption of invalid key returns None."""
    result = claude_client._decrypt_api_key_from_db("invalid-encrypted-key")
    # Should return None since decryption will fail
    assert result is None


def test_claude_client_model_selection(mock_orchestrator):
    """Test model selection from config."""
    mock_orchestrator.config.claude_model = "claude-3-opus-20240229"
    with patch("socratic_nexus.clients.claude_client.anthropic"):
        client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
        assert client.model == "claude-3-opus-20240229"


def test_claude_client_default_model():
    """Test default model when no orchestrator config."""
    with patch("socratic_nexus.clients.claude_client.anthropic"):
        client = ClaudeClient(api_key="test-key", orchestrator=None)
        # Should use a default value
        assert client.model is not None


def test_claude_client_can_use_optional_api_key():
    """Test that ClaudeClient can be initialized without API key."""
    with patch("socratic_nexus.clients.claude_client.anthropic"):
        client = ClaudeClient(api_key=None, orchestrator=None)
        assert client.api_key is None


def test_project_context_attribute_access():
    """Test accessing ProjectContext attributes."""
    context = ProjectContext(
        project_name="Test",
        phase="implementation",
        goals=["Goal 1"],
        tech_stack=["Python"],
    )
    assert context.project_name == "Test"
    assert context.phase == "implementation"
    assert context.goals == ["Goal 1"]
    assert context.tech_stack == ["Python"]


def test_conflict_info_in_client_context(claude_client):
    """Test ConflictInfo usage in client."""
    conflict = ConflictInfo(
        description="Test conflict",
        file_path="test.py",
        line_number=10,
    )
    assert conflict.description == "Test conflict"
    assert conflict.file_path == "test.py"
    assert conflict.line_number == 10


def test_token_usage_cost_calculation():
    """Test token cost calculation in client context."""
    # The cost should be calculated by the client
    # This test verifies the model can store cost information
    usage_with_cost = TokenUsage(
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        provider="claude",
        model="claude-3-sonnet",
        cost_usd=0.015,
    )
    assert usage_with_cost.cost_usd == 0.015
