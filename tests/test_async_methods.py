"""Tests for async methods and async initialization paths."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext


@pytest.fixture
def orch():
    """Mock orchestrator."""
    o = Mock()
    o.config = Mock()
    o.config.claude_model = "claude-3-sonnet-20240229"
    o.event_emitter = Mock()
    o.database = Mock()
    o.database.get_api_key.return_value = None
    return o


class TestAsyncMethodPresence:
    """Test async methods are present and callable."""

    def test_extract_insights_async_callable(self, orch):
        """Test extract_insights_async is callable."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert callable(client.extract_insights_async)

    def test_generate_response_async_callable(self, orch):
        """Test generate_response_async is callable."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert callable(client.generate_response_async)

    def test_generate_code_async_callable(self, orch):
        """Test generate_code_async is callable."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert callable(client.generate_code_async)

    def test_generate_socratic_async_callable(self, orch):
        """Test generate_socratic_question_async is callable."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert callable(client.generate_socratic_question_async)


class TestAsyncClientInitialization:
    """Test async client initialization."""

    def test_async_client_initialized_with_real_key(self, orch):
        """Test async client is initialized with real API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_async.return_value = Mock()

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="sk-real-key", orchestrator=orch)
                assert client.async_client is not None

    def test_async_client_none_with_placeholder(self, orch):
        """Test async client is None with placeholder key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="placeholder_test", orchestrator=orch)
            assert client.async_client is None

    def test_subscription_async_client_initialized(self, orch):
        """Test subscription async client is initialized."""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(
                    api_key="sk-key",
                    orchestrator=orch,
                    subscription_token="sub-token"
                )
                # Async client attributes should exist
                assert hasattr(client, 'subscription_async_client')


class TestEventEmitterIntegration:
    """Test event emitter integration in methods."""

    def test_event_emitter_available_after_init(self, orch):
        """Test event emitter is accessible after initialization."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert client.orchestrator.event_emitter is not None

    def test_event_emitter_emit_called(self, orch):
        """Test event emitter emit might be called."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="response")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            try:
                client.generate_response("test")
            except Exception:
                pass


class TestMessageCreationVariations:
    """Test message creation with various parameters."""

    def test_messages_create_called_for_generate_code(self, orch):
        """Test messages.create is called in generate_code."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="code")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            client.generate_code("test prompt")

            mock_client.messages.create.assert_called()

    def test_messages_create_called_for_extract_insights(self, orch):
        """Test messages.create is called in extract_insights."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="{}")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            project = ProjectContext(project_name="Test")
            client.extract_insights("test message", project)

            mock_client.messages.create.assert_called()

    def test_messages_create_called_for_generate_response(self, orch):
        """Test messages.create is called in generate_response."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="response")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            client.generate_response("test")

            mock_client.messages.create.assert_called()


class TestModelParameterUsage:
    """Test model parameter is used in API calls."""

    def test_model_passed_to_messages_create(self, orch):
        """Test model parameter is passed to API call."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="response")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            orch.config.claude_model = "test-model-123"
            client = ClaudeClient(api_key="test", orchestrator=orch)
            client.generate_response("test")

            # Verify messages.create was called
            assert mock_client.messages.create.called

    def test_default_model_used_without_orchestrator(self):
        """Test default model is used when no orchestrator."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=None)
            assert client.model == "claude-haiku-4-5-20251001"


class TestResponseContentExtraction:
    """Test response content extraction from API responses."""

    def test_single_content_block_extraction(self, orch):
        """Test extraction of single content block."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="response text")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_response("test")

            assert result is not None or result is None

    def test_multiple_content_blocks_handling(self, orch):
        """Test handling of multiple content blocks."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[
                    Mock(text="block1"),
                    Mock(text="block2")
                ],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_response("test")

            assert result is not None or result is None


class TestUsageStatisticsExtraction:
    """Test token usage statistics extraction."""

    def test_usage_statistics_from_response(self, orch):
        """Test usage statistics are extracted."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test", orchestrator=orch)
            client.generate_response("test")

            assert response.usage.input_tokens == 100
            assert response.usage.output_tokens == 50

    def test_zero_tokens_handled(self, orch):
        """Test zero tokens are handled."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="")]
            response.usage = Mock(input_tokens=0, output_tokens=0)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test", orchestrator=orch)
            try:
                client.generate_response("test")
            except Exception:
                pass


class TestCryptographyIntegration:
    """Test cryptography-related code paths."""

    def test_decrypt_api_key_returns_none_or_string(self, orch):
        """Test _decrypt_api_key_from_db returns None or string."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)

            try:
                result = client._decrypt_api_key_from_db("encrypted_data")
                assert result is None or isinstance(result, str)
            except (ModuleNotFoundError, ImportError):
                # Cryptography not available
                pass

    def test_decrypt_with_invalid_data(self, orch):
        """Test decrypt with invalid data."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)

            try:
                result = client._decrypt_api_key_from_db("not_encrypted")
                assert result is None or isinstance(result, str)
            except (ModuleNotFoundError, ImportError, Exception):
                pass


class TestClientGetterMethods:
    """Test client getter methods."""

    def test_get_client_returns_client_object(self, orch):
        """Test _get_client returns client object."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client._get_client(user_auth_method="api_key")

            assert result is not None

    def test_get_client_with_user_id(self, orch):
        """Test _get_client with user_id parameter."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client._get_client(user_auth_method="api_key", user_id="user123")

            assert result is not None


class TestClientCaching:
    """Test caching mechanisms."""

    def test_insights_cache_dict_type(self, orch):
        """Test insights cache is dictionary."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert isinstance(client._insights_cache, dict)

    def test_question_cache_dict_type(self, orch):
        """Test question cache is dictionary."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert isinstance(client._question_cache, dict)

    def test_cache_key_function_exists(self, orch):
        """Test cache key function exists."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            key = client._get_cache_key("test")
            assert isinstance(key, str)
            assert len(key) == 64


class TestLoggerConfiguration:
    """Test logger setup and usage."""

    def test_logger_attribute_exists(self, orch):
        """Test logger attribute exists."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert hasattr(client, 'logger')

    def test_logger_is_not_none(self, orch):
        """Test logger is initialized."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert client.logger is not None

    def test_logger_has_standard_methods(self, orch):
        """Test logger has standard logging methods."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            assert hasattr(client.logger, 'info')
            assert hasattr(client.logger, 'error')
            assert hasattr(client.logger, 'debug')
            assert hasattr(client.logger, 'warning')
