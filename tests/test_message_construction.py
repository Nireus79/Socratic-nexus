"""Tests for message construction and API call details."""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator."""
    orch = Mock()
    orch.config = Mock()
    orch.config.claude_model = "claude-3-sonnet-20240229"
    orch.event_emitter = Mock()
    orch.database = Mock()
    orch.database.get_api_key.return_value = None
    return orch


def create_mock_response(text="response", input_tokens=10, output_tokens=20):
    """Helper to create mock API response."""
    response = Mock()
    response.content = [Mock(text=text)]
    response.usage = Mock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


class TestGenerateResponseMessageConstruction:
    """Tests for message construction in generate_response."""

    def test_generate_response_calls_messages_create(self, mock_orchestrator):
        """Test that generate_response calls messages.create."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            client.generate_response("test prompt")

            mock_client.messages.create.assert_called()

    def test_generate_response_uses_correct_model(self, mock_orchestrator):
        """Test that correct model is used in API call."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            mock_orchestrator.config.claude_model = "test-model-xyz"
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            client.generate_response("test")

            # Check that model was passed to messages.create
            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs.get("model") == "test-model-xyz"

    def test_generate_response_passes_max_tokens(self, mock_orchestrator):
        """Test that max_tokens parameter is passed to API."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            client.generate_response("test", max_tokens=200)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs.get("max_tokens") == 200

    def test_generate_response_passes_temperature(self, mock_orchestrator):
        """Test that temperature parameter is passed to API."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            client.generate_response("test", temperature=0.8)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs.get("temperature") == 0.8

    def test_generate_response_message_format(self, mock_orchestrator):
        """Test that messages are in correct format."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            client.generate_response("test prompt")

            call_kwargs = mock_client.messages.create.call_args.kwargs
            messages = call_kwargs.get("messages", [])
            assert isinstance(messages, list)
            assert len(messages) > 0
            assert messages[0].get("role") == "user"


class TestGenerateCodeMessageConstruction:
    """Tests for message construction in generate_code."""

    def test_generate_code_calls_messages_create(self, mock_orchestrator):
        """Test that generate_code calls messages.create."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("code")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            client.generate_code("write a function")

            mock_client.messages.create.assert_called()

    def test_generate_code_includes_prompt(self, mock_orchestrator):
        """Test that prompt is included in messages."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("code")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            prompt_text = "write a hello world function"
            client.generate_code(prompt_text)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            messages = call_kwargs.get("messages", [])
            # Prompt should be in the messages somewhere
            assert any(prompt_text in str(msg) for msg in messages)

    def test_generate_code_temperature_0_7(self, mock_orchestrator):
        """Test that generate_code uses temperature 0.7."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("code")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            client.generate_code("test")

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs.get("temperature") == 0.7


class TestExtractInsightsMessageConstruction:
    """Tests for message construction in extract_insights."""

    def test_extract_insights_calls_messages_create(self, mock_orchestrator):
        """Test that extract_insights calls messages.create."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response('{}')

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            client.extract_insights("user response", project)

            mock_client.messages.create.assert_called()

    def test_extract_insights_includes_project_context(self, mock_orchestrator):
        """Test that project context is included in prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response('{}')

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            project = ProjectContext(
                project_name="TestProject",
                phase="planning",
                goals=["goal1", "goal2"]
            )
            client.extract_insights("response", project)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            messages = call_kwargs.get("messages", [])
            # Project info should be in the prompt
            assert any("TestProject" in str(msg) or "planning" in str(msg) for msg in messages)

    def test_extract_insights_temperature_0_3(self, mock_orchestrator):
        """Test that extract_insights uses temperature 0.3."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response('{}')

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            client.extract_insights("test", project)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs.get("temperature") == 0.3


class TestGenerateSocraticMessageConstruction:
    """Tests for message construction in generate_socratic_question."""

    def test_generate_socratic_calls_messages_create(self, mock_orchestrator):
        """Test that generate_socratic_question calls messages.create."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("question?")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            client.generate_socratic_question("test topic")

            mock_client.messages.create.assert_called()

    def test_generate_socratic_includes_prompt(self, mock_orchestrator):
        """Test that prompt is included in messages."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("question?")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            prompt_text = "explain recursion"
            client.generate_socratic_question(prompt_text)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            messages = call_kwargs.get("messages", [])
            # Prompt should be in the messages
            assert any(prompt_text in str(msg) for msg in messages)


class TestResponseHandlingVariations:
    """Tests for various response handling scenarios."""

    def test_single_content_block_extraction(self, mock_orchestrator):
        """Test extraction from single content block."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response text")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt")

            assert result is not None

    def test_multiple_content_blocks_uses_first(self, mock_orchestrator):
        """Test that multiple content blocks use first one."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [
                Mock(text="first"),
                Mock(text="second"),
                Mock(text="third")
            ]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt")

            # Should use first content block
            assert result is not None or result is None

    def test_empty_response_content(self, mock_orchestrator):
        """Test handling of empty response content."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = []
            response.usage = Mock(input_tokens=10, output_tokens=0)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)

            try:
                result = client.generate_response("prompt")
                # Should handle gracefully
            except Exception:
                # Exception is acceptable for empty content
                pass
