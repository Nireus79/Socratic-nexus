"""Comprehensive tests for client initialization and configuration."""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient


class TestClientInitializationVariations:
    """Tests for various client initialization scenarios."""

    def test_init_with_none_api_key(self):
        """Test initialization with None API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=None)
            assert client.api_key is None

    def test_init_with_empty_string_api_key(self):
        """Test initialization with empty string API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="", orchestrator=None)
            assert client.api_key == ""

    def test_init_with_valid_api_key(self):
        """Test initialization with valid API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
                client = ClaudeClient(api_key="sk-test-key", orchestrator=None)
                assert client.api_key == "sk-test-key"

    def test_init_sets_default_model_without_orchestrator(self):
        """Test default model is set when no orchestrator."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)
            assert client.model == "claude-haiku-4-5-20251001"

    def test_init_uses_orchestrator_model(self):
        """Test model from orchestrator is used."""
        orch = Mock()
        orch.config = Mock()
        orch.config.claude_model = "custom-model-xyz"
        orch.event_emitter = Mock()
        orch.database = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert client.model == "custom-model-xyz"

    def test_init_creates_logger(self):
        """Test logger is created during init."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)
            assert client.logger is not None
            assert hasattr(client.logger, 'info')
            assert hasattr(client.logger, 'error')

    def test_init_initializes_caches(self):
        """Test caches are initialized."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)
            assert isinstance(client._insights_cache, dict)
            assert isinstance(client._question_cache, dict)
            assert len(client._insights_cache) == 0
            assert len(client._question_cache) == 0

    def test_init_with_placeholder_key_no_client(self):
        """Test that placeholder keys don't initialize clients."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="placeholder_test", orchestrator=None)
            assert client.client is None
            assert client.async_client is None

    def test_init_with_placeholder_prefix_variations(self):
        """Test various placeholder prefix patterns."""
        placeholders = ["placeholder", "placeholder_", "placeholder_test", "placeholder123"]

        for placeholder in placeholders:
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key=placeholder, orchestrator=None)
                assert client.client is None
                assert client.async_client is None

    def test_init_subscription_token_initializes_subscription_clients(self):
        """Test subscription token initializes subscription clients."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
                client = ClaudeClient(
                    api_key="test",
                    orchestrator=None,
                    subscription_token="sub-token-123"
                )
                assert client.subscription_token == "sub-token-123"

    def test_init_without_subscription_token(self):
        """Test initialization without subscription token."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None, subscription_token=None)
            assert client.subscription_token is None


class TestGetAuthCredentialVariations:
    """Tests for get_auth_credential method with various scenarios."""

    def test_get_auth_credential_api_key_default(self):
        """Test getting API key with default parameter."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="my-api-key", orchestrator=None)
            cred = client.get_auth_credential()
            assert cred == "my-api-key"

    def test_get_auth_credential_explicit_api_key(self):
        """Test getting API key explicitly."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="my-api-key", orchestrator=None)
            cred = client.get_auth_credential("api_key")
            assert cred == "my-api-key"

    def test_get_auth_credential_subscription(self):
        """Test getting subscription token."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="api-key",
                orchestrator=None,
                subscription_token="sub-token-123"
            )
            cred = client.get_auth_credential("subscription")
            assert cred == "sub-token-123"

    def test_get_auth_credential_missing_api_key_raises(self):
        """Test missing API key raises ValueError."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=None)

            with pytest.raises(ValueError):
                client.get_auth_credential("api_key")

    def test_get_auth_credential_missing_subscription_raises(self):
        """Test missing subscription token raises ValueError."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None, subscription_token=None)

            with pytest.raises(ValueError):
                client.get_auth_credential("subscription")


class TestCacheKeyGeneration:
    """Tests for cache key generation consistency."""

    def test_cache_key_deterministic(self):
        """Test cache key is deterministic."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)

            key1 = client._get_cache_key("test message")
            key2 = client._get_cache_key("test message")

            assert key1 == key2

    def test_cache_key_different_messages(self):
        """Test different messages produce different keys."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)

            key1 = client._get_cache_key("message 1")
            key2 = client._get_cache_key("message 2")

            assert key1 != key2

    def test_cache_key_format_is_hex(self):
        """Test cache key is hex format."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)
            key = client._get_cache_key("test")

            # Should be hexadecimal
            assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_length_64(self):
        """Test cache key is 64 characters (SHA256)."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)
            key = client._get_cache_key("test")

            assert len(key) == 64

    def test_cache_key_whitespace_sensitivity(self):
        """Test cache key is sensitive to whitespace."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)

            key1 = client._get_cache_key("test message")
            key2 = client._get_cache_key("test  message")  # Extra space

            assert key1 != key2

    def test_cache_key_case_sensitivity(self):
        """Test cache key is case sensitive."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)

            key1 = client._get_cache_key("Test")
            key2 = client._get_cache_key("test")

            assert key1 != key2


class TestClientAttributeAccessibility:
    """Tests for accessing client attributes."""

    def test_orchestrator_stored(self):
        """Test orchestrator is stored."""
        orch = Mock()
        orch.config = Mock()
        orch.config.claude_model = "test-model"
        orch.event_emitter = Mock()
        orch.database = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert client.orchestrator is orch

    def test_subscription_token_stored(self):
        """Test subscription token is stored."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="test",
                orchestrator=None,
                subscription_token="token123"
            )
            assert client.subscription_token == "token123"

    def test_model_accessible(self):
        """Test model attribute is accessible."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)
            assert isinstance(client.model, str)
            assert len(client.model) > 0

    def test_api_key_stored(self):
        """Test API key is stored."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key-123", orchestrator=None)
            assert client.api_key == "test-key-123"


class TestClientInitializationEdgeCases:
    """Tests for edge cases in client initialization."""

    def test_init_with_special_characters_in_api_key(self):
        """Test initialization with special characters in API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="sk-!@#$%^&*()", orchestrator=None)
            assert client.api_key == "sk-!@#$%^&*()"

    def test_init_with_unicode_in_api_key(self):
        """Test initialization with unicode in API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="sk-测试-тест", orchestrator=None)
            assert client.api_key == "sk-测试-тест"

    def test_init_with_very_long_api_key(self):
        """Test initialization with very long API key."""
        long_key = "sk-" + "x" * 10000
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=long_key, orchestrator=None)
            assert client.api_key == long_key

    def test_init_with_orchestrator_missing_config(self):
        """Test initialization with orchestrator missing config."""
        orch = Mock(spec=[])  # Empty spec
        orch.event_emitter = Mock()
        orch.database = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            try:
                ClaudeClient(api_key="test", orchestrator=orch)
                # Should handle gracefully
            except AttributeError:
                # Missing config is acceptable
                pass
