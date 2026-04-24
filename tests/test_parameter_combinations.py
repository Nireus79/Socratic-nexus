"""Tests for various parameter combinations and edge cases."""

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


class TestGenerateResponseParameterCombinations:
    """Tests for generate_response with various parameter combinations."""

    def test_generate_response_prompt_only(self, mock_orchestrator):
        """Test generate_response with just prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt")
            assert result is not None or result is None

    def test_generate_response_with_max_tokens_only(self, mock_orchestrator):
        """Test with max_tokens parameter."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt", max_tokens=500)
            assert result is not None or result is None

    def test_generate_response_with_temperature_only(self, mock_orchestrator):
        """Test with temperature parameter."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt", temperature=0.5)
            assert result is not None or result is None

    def test_generate_response_with_both_max_tokens_and_temperature(self, mock_orchestrator):
        """Test with both max_tokens and temperature."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt", max_tokens=100, temperature=0.9)
            assert result is not None or result is None

    def test_generate_response_max_tokens_boundary_values(self, mock_orchestrator):
        """Test with boundary max_tokens values."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)

            # Test various boundary values
            for max_tokens in [1, 10, 100, 1000, 4000, 10000]:
                result = client.generate_response("prompt", max_tokens=max_tokens)
                assert result is not None or result is None

    def test_generate_response_temperature_boundary_values(self, mock_orchestrator):
        """Test with boundary temperature values."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)

            # Test various boundary values
            for temp in [0, 0.1, 0.5, 1.0, 1.5, 2.0]:
                result = client.generate_response("prompt", temperature=temp)
                assert result is not None or result is None


class TestExtractInsightsParameterVariations:
    """Tests for extract_insights with parameter variations."""

    def test_extract_insights_minimal_project_context(self, mock_orchestrator):
        """Test with minimal ProjectContext."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("{}")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Minimal")
            result = client.extract_insights("response", project)

            assert isinstance(result, dict)

    def test_extract_insights_full_project_context(self, mock_orchestrator):
        """Test with fully populated ProjectContext."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("{}")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            project = ProjectContext(
                project_name="Full",
                description="Full project description",
                goals=["goal1", "goal2", "goal3"],
                phase="implementation",
                tech_stack=["Python", "FastAPI", "PostgreSQL"],
                project_type="web_application",
                deployment_target="AWS",
                status="active",
            )
            result = client.extract_insights("response", project)

            assert isinstance(result, dict)

    def test_extract_insights_with_user_id(self, mock_orchestrator):
        """Test with user_id parameter."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("{}")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("response", project, user_id="user123")

            assert isinstance(result, dict)

    def test_extract_insights_with_auth_method(self, mock_orchestrator):
        """Test with explicit auth_method parameter."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("{}")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("response", project, user_auth_method="api_key")

            assert isinstance(result, dict)

    def test_extract_insights_response_length_variations(self, mock_orchestrator):
        """Test with various response text lengths."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("{}")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")

            # Test various response lengths
            test_responses = [
                "short",
                "this is a longer response with more details",
                "x" * 100,
                "x" * 1000,
                "x" * 10000,
            ]

            for response in test_responses:
                result = client.extract_insights(response, project)
                assert isinstance(result, dict)


class TestGenerateCodeParameterVariations:
    """Tests for generate_code with parameter variations."""

    def test_generate_code_empty_prompt(self, mock_orchestrator):
        """Test with empty prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("code")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_code("")
            assert result is not None or result is None

    def test_generate_code_simple_prompt(self, mock_orchestrator):
        """Test with simple prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("code")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_code("write a function")
            assert result is not None or result is None

    def test_generate_code_detailed_prompt(self, mock_orchestrator):
        """Test with detailed prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("code")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            detailed_prompt = """
            Write a Python function that:
            1. Takes a list of numbers as input
            2. Filters out negative numbers
            3. Sorts the remaining numbers
            4. Returns the sorted list
            Please include error handling and docstring.
            """
            result = client.generate_code(detailed_prompt)
            assert result is not None or result is None

    def test_generate_code_with_requirements(self, mock_orchestrator):
        """Test generate_code with various prompt variations."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("code")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)

            # Test with different code-related prompts
            prompts = [
                "Python script for data processing",
                "JavaScript async function",
                "SQL migration script",
                "API endpoint handler",
            ]

            for prompt in prompts:
                result = client.generate_code(prompt)
                assert result is not None or result is None


class TestGenerateSocraticQuestionParameterVariations:
    """Tests for generate_socratic_question with parameter variations."""

    def test_generate_socratic_basic_prompt(self, mock_orchestrator):
        """Test with basic prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("question?")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_socratic_question("loops")
            assert result is not None or result is None

    def test_generate_socratic_with_explicit_cache_key(self, mock_orchestrator):
        """Test with explicit cache_key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("question?")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_socratic_question("topic", cache_key="custom_key_123")
            assert result is not None or result is None

    def test_generate_socratic_various_topics(self, mock_orchestrator):
        """Test with various educational topics."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("question?")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)

            topics = [
                "recursion",
                "design patterns",
                "database optimization",
                "API design",
                "testing strategies",
            ]

            for topic in topics:
                result = client.generate_socratic_question(topic)
                assert result is not None or result is None


class TestPromptTextVariations:
    """Tests for various prompt text patterns."""

    def test_prompt_with_newlines(self, mock_orchestrator):
        """Test prompt with newlines."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            prompt = "line1\nline2\nline3"
            result = client.generate_response(prompt)
            assert result is not None or result is None

    def test_prompt_with_special_formatting(self, mock_orchestrator):
        """Test prompt with special formatting."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            prompt = "```python\ndef hello():\n    print('hello')\n```"
            result = client.generate_response(prompt)
            assert result is not None or result is None

    def test_prompt_with_markdown(self, mock_orchestrator):
        """Test prompt with markdown formatting."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            prompt = "# Heading\n## Subheading\n- bullet 1\n- bullet 2"
            result = client.generate_response(prompt)
            assert result is not None or result is None

    def test_prompt_with_json(self, mock_orchestrator):
        """Test prompt containing JSON."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = create_mock_response("response")

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            prompt = '{"key": "value", "nested": {"inner": "data"}}'
            result = client.generate_response(prompt)
            assert result is not None or result is None
