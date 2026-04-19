"""
Tests for Openclaw Integration.

Tests for NexusLLMSkill - Openclaw integration with Socrates Nexus.
"""

import pytest
from unittest.mock import Mock, patch
from socratic_nexus.integrations.openclaw import NexusLLMSkill
from socratic_nexus import ChatResponse, TokenUsage, RateLimitError


class TestNexusLLMSkillInitialization:
    """Test NexusLLMSkill initialization."""

    def test_initialization_defaults(self):
        """Test skill initializes with default parameters."""
        skill = NexusLLMSkill()
        assert skill.config.provider == "anthropic"
        assert skill.config.model == "claude-opus"
        assert skill.config.temperature == 0.7
        assert skill.config.retry_attempts == 3
        assert skill.config.cache_responses is True

    def test_initialization_custom_provider(self):
        """Test skill initializes with custom provider."""
        skill = NexusLLMSkill(provider="openai", model="gpt-4")
        assert skill.config.provider == "openai"
        assert skill.config.model == "gpt-4"

    def test_initialization_custom_parameters(self):
        """Test skill initializes with custom parameters."""
        skill = NexusLLMSkill(
            provider="google",
            model="gemini-pro",
            temperature=0.5,
            max_tokens=512,
            retry_attempts=5,
            cache_responses=False,
        )
        assert skill.config.provider == "google"
        assert skill.config.model == "gemini-pro"
        assert skill.config.temperature == 0.5
        assert skill.config.max_tokens == 512
        assert skill.config.retry_attempts == 5
        assert skill.config.cache_responses is False

    def test_initialization_with_api_key(self):
        """Test skill initializes with API key."""
        skill = NexusLLMSkill(api_key="test-key-123")
        assert skill.config.api_key == "test-key-123"

    def test_client_created_on_init(self):
        """Test that LLMClient is created during initialization."""
        skill = NexusLLMSkill()
        assert skill.client is not None
        assert hasattr(skill.client, "chat")


class TestNexusLLMSkillQuery:
    """Test NexusLLMSkill query method."""

    def test_query_basic(self):
        """Test basic query execution."""
        skill = NexusLLMSkill()
        with patch.object(skill.client, "chat") as mock_chat:
            mock_response = ChatResponse(
                provider="anthropic",
                model="claude-opus",
                content="Test response",
                usage=TokenUsage(
                    input_tokens=10,
                    output_tokens=20,
                    total_tokens=10 + 20,
                    provider="anthropic",
                    model="claude-opus",
                ),
            )
            mock_chat.return_value = mock_response

            response = skill.query("What is AI?")

            assert response.content == "Test response"
            assert response.usage.total_tokens == 30
            mock_chat.assert_called_once_with("What is AI?")

    def test_query_with_kwargs(self):
        """Test query with additional kwargs."""
        skill = NexusLLMSkill()
        with patch.object(skill.client, "chat") as mock_chat:
            mock_response = ChatResponse(
                provider="anthropic",
                model="claude-opus",
                content="Response",
                usage=TokenUsage(
                    input_tokens=5,
                    output_tokens=15,
                    total_tokens=5 + 15,
                    provider="anthropic",
                    model="claude-opus",
                ),
            )
            mock_chat.return_value = mock_response

            response = skill.query("Tell me", temperature=0.5, max_tokens=100)

            assert response.content == "Response"
            mock_chat.assert_called_once_with("Tell me", temperature=0.5, max_tokens=100)

    def test_query_error_handling(self):
        """Test query handles errors properly."""
        skill = NexusLLMSkill()
        with patch.object(skill.client, "chat") as mock_chat:
            mock_chat.side_effect = RateLimitError("Rate limit exceeded", retry_after=60)

            with pytest.raises(RateLimitError):
                skill.query("Query")


class TestNexusLLMSkillStreaming:
    """Test NexusLLMSkill streaming."""

    def test_stream_basic(self):
        """Test streaming query."""
        skill = NexusLLMSkill()
        with patch.object(skill.client, "stream") as mock_stream:
            mock_response = ChatResponse(
                provider="anthropic",
                model="claude-opus",
                content="Streamed response",
                usage=TokenUsage(
                    input_tokens=5,
                    output_tokens=25,
                    total_tokens=5 + 25,
                    provider="anthropic",
                    model="claude-opus",
                ),
            )
            mock_stream.return_value = mock_response

            on_chunk = Mock()
            response = skill.stream("Write a poem", on_chunk=on_chunk)

            assert response.content == "Streamed response"
            mock_stream.assert_called_once_with("Write a poem", on_chunk=on_chunk)

    def test_stream_with_callback(self):
        """Test streaming with callback function."""
        skill = NexusLLMSkill()

        def chunk_callback(chunk):
            pass

        with patch.object(skill.client, "stream") as mock_stream:
            mock_response = ChatResponse(
                provider="anthropic",
                model="claude-opus",
                content="Result",
                usage=TokenUsage(
                    input_tokens=5,
                    output_tokens=15,
                    total_tokens=5 + 15,
                    provider="anthropic",
                    model="claude-opus",
                ),
            )
            mock_stream.return_value = mock_response

            response = skill.stream("Query", on_chunk=chunk_callback)
            assert response is not None


class TestNexusLLMSkillProviderSwitching:
    """Test provider switching functionality."""

    def test_switch_provider(self):
        """Test switching to different provider."""
        skill = NexusLLMSkill(provider="anthropic", model="claude-opus")
        assert skill.config.provider == "anthropic"

        skill.switch_provider("openai", "gpt-4")

        assert skill.config.provider == "openai"
        assert skill.config.model == "gpt-4"
        assert skill.client is not None

    def test_switch_provider_with_api_key(self):
        """Test switching provider with new API key."""
        skill = NexusLLMSkill(provider="anthropic", api_key="old-key")
        skill.switch_provider("openai", "gpt-4", api_key="new-key")

        assert skill.config.provider == "openai"
        assert skill.config.model == "gpt-4"
        assert skill.config.api_key == "new-key"

    def test_switch_provider_maintains_other_settings(self):
        """Test that switching provider maintains other settings."""
        skill = NexusLLMSkill(
            provider="anthropic",
            temperature=0.5,
            max_tokens=256,
            retry_attempts=5,
        )
        skill.switch_provider("openai", "gpt-4")

        assert skill.config.temperature == 0.5
        assert skill.config.max_tokens == 256
        assert skill.config.retry_attempts == 5

    def test_multiple_provider_switches(self):
        """Test switching between multiple providers."""
        skill = NexusLLMSkill()

        skill.switch_provider("openai", "gpt-4")
        assert skill.config.provider == "openai"

        skill.switch_provider("google", "gemini-pro")
        assert skill.config.provider == "google"

        skill.switch_provider("anthropic", "claude-opus")
        assert skill.config.provider == "anthropic"


class TestNexusLLMSkillUsageTracking:
    """Test usage statistics tracking."""

    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        skill = NexusLLMSkill()

        with patch.object(skill.client, "get_usage_stats") as mock_stats:
            mock_stats.return_value = Mock(
                total_requests=5,
                total_input_tokens=100,
                total_output_tokens=200,
                total_cost_usd=1.50,
                by_provider={"anthropic": {"requests": 5, "cost_usd": 1.50}},
                by_model={"claude-opus": {"requests": 5, "cost_usd": 1.50}},
            )

            stats = skill.get_usage_stats()

            assert stats["total_requests"] == 5
            assert stats["total_input_tokens"] == 100
            assert stats["total_output_tokens"] == 200
            assert stats["total_cost_usd"] == 1.50

    def test_usage_stats_structure(self):
        """Test that usage stats have correct structure."""
        skill = NexusLLMSkill()

        with patch.object(skill.client, "get_usage_stats") as mock_stats:
            mock_stats.return_value = Mock(
                total_requests=1,
                total_input_tokens=10,
                total_output_tokens=20,
                total_cost_usd=0.10,
                by_provider={},
                by_model={},
            )

            stats = skill.get_usage_stats()

            assert "total_requests" in stats
            assert "total_input_tokens" in stats
            assert "total_output_tokens" in stats
            assert "total_cost_usd" in stats
            assert "by_provider" in stats
            assert "by_model" in stats


class TestNexusLLMSkillCallbacks:
    """Test callback functionality."""

    def test_add_usage_callback(self):
        """Test adding usage callback."""
        skill = NexusLLMSkill(api_key="test-key")
        callback = Mock()

        with patch.object(skill.client, "add_usage_callback") as mock_add:
            skill.add_usage_callback(callback)
            mock_add.assert_called_once_with(callback)

    def test_multiple_callbacks(self):
        """Test adding multiple callbacks."""
        skill = NexusLLMSkill()
        callback1 = Mock()
        callback2 = Mock()

        with patch.object(skill.client, "add_usage_callback"):
            skill.add_usage_callback(callback1)
            skill.add_usage_callback(callback2)

    def test_callback_receives_usage(self):
        """Test that callback receives TokenUsage object."""
        skill = NexusLLMSkill()
        callback = Mock()

        with patch.object(skill.client, "add_usage_callback"):
            skill.add_usage_callback(callback)


class TestNexusLLMSkillRepresentation:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        skill = NexusLLMSkill(provider="anthropic", model="claude-opus")
        repr_str = repr(skill)

        assert "NexusLLMSkill" in repr_str
        assert "anthropic" in repr_str
        assert "claude-opus" in repr_str

    def test_repr_with_different_provider(self):
        """Test __repr__ with different provider."""
        skill = NexusLLMSkill(provider="openai", model="gpt-4")
        repr_str = repr(skill)

        assert "NexusLLMSkill" in repr_str
        assert "openai" in repr_str
        assert "gpt-4" in repr_str


class TestNexusLLMSkillConfiguration:
    """Test configuration management."""

    def test_temperature_configuration(self):
        """Test temperature configuration."""
        skill = NexusLLMSkill(temperature=0.9)
        assert skill.config.temperature == 0.9

    def test_max_tokens_configuration(self):
        """Test max_tokens configuration."""
        skill = NexusLLMSkill(max_tokens=2048)
        assert skill.config.max_tokens == 2048

    def test_retry_attempts_configuration(self):
        """Test retry_attempts configuration."""
        skill = NexusLLMSkill(retry_attempts=5)
        assert skill.config.retry_attempts == 5

    def test_cache_configuration(self):
        """Test cache configuration."""
        skill = NexusLLMSkill(cache_responses=False, cache_ttl=600)
        assert skill.config.cache_responses is False
        assert skill.config.cache_ttl == 600


class TestNexusLLMSkillEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_response_content(self):
        """Test handling of empty response content."""
        skill = NexusLLMSkill()
        with patch.object(skill.client, "chat") as mock_chat:
            mock_response = ChatResponse(
                provider="anthropic",
                model="claude-opus",
                content="",
                usage=TokenUsage(
                    input_tokens=5,
                    output_tokens=0,
                    total_tokens=5 + 0,
                    provider="anthropic",
                    model="claude-opus",
                ),
            )
            mock_chat.return_value = mock_response

            response = skill.query("Query")
            assert response.content == ""

    def test_very_long_response(self):
        """Test handling of very long responses."""
        skill = NexusLLMSkill()
        long_content = "A" * 10000
        with patch.object(skill.client, "chat") as mock_chat:
            mock_response = ChatResponse(
                provider="anthropic",
                model="claude-opus",
                content=long_content,
                usage=TokenUsage(
                    input_tokens=5,
                    output_tokens=5000,
                    total_tokens=5 + 5000,
                    provider="anthropic",
                    model="claude-opus",
                ),
            )
            mock_chat.return_value = mock_response

            response = skill.query("Query")
            assert len(response.content) == 10000

    def test_zero_tokens(self):
        """Test handling of zero tokens."""
        skill = NexusLLMSkill()
        with patch.object(skill.client, "chat") as mock_chat:
            mock_response = ChatResponse(
                provider="anthropic",
                model="claude-opus",
                content="Response",
                usage=TokenUsage(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0 + 0,
                    provider="anthropic",
                    model="claude-opus",
                ),
            )
            mock_chat.return_value = mock_response

            response = skill.query("Query")
            assert response.usage.total_tokens == 0
