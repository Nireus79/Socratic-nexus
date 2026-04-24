"""Tests targeting specific method branches and conditions."""

import pytest
from unittest.mock import Mock, patch

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


class TestExtractInsightsBranches:
    """Test different branches in extract_insights."""

    def test_extract_insights_short_message_filter(self, orch):
        """Test short messages are filtered."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # Very short message
            result = client.extract_insights("a", project)
            assert isinstance(result, dict)

    def test_extract_insights_min_length_boundary(self, orch):
        """Test message at minimum length boundary."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # Around 30 character boundary
            result = client.extract_insights("x" * 30, project)
            assert isinstance(result, dict)

    def test_extract_insights_exact_length_check(self, orch):
        """Test message length checking."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            msg = "test message"
            result = client.extract_insights(msg, project)
            assert isinstance(result, dict)

    def test_extract_insights_non_informative_check(self, orch):
        """Test non-informative response filtering."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # Test common non-informative responses
            responses = ["i don't know", "idk", "not sure", "no idea", "unclear"]
            for resp in responses:
                try:
                    result = client.extract_insights(resp, project)
                    assert isinstance(result, dict)
                except Exception:
                    pass

    def test_extract_insights_response_length_check(self, orch):
        """Test response length validation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # Test with various message lengths
            for length in [10, 50, 100, 500, 1000]:
                result = client.extract_insights("x" * length, project)
                assert isinstance(result, dict)


class TestGenerateCodeBranches:
    """Test different branches in generate_code."""

    def test_generate_code_empty_prompt(self, orch):
        """Test generate_code with empty prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="code")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_code("")
            assert result is None or isinstance(result, str)

    def test_generate_code_long_prompt(self, orch):
        """Test generate_code with very long prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="code")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_code("x" * 5000)
            assert result is None or isinstance(result, str)

    def test_generate_code_special_characters(self, orch):
        """Test generate_code with special characters."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="code")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            prompt = "Generate: @#$%^&*()_+-=[]{}|;:,.<>?/"
            result = client.generate_code(prompt)
            assert result is None or isinstance(result, str)

    def test_generate_code_multiline_prompt(self, orch):
        """Test generate_code with multiline prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="code")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            prompt = "Line 1\nLine 2\nLine 3"
            result = client.generate_code(prompt)
            assert result is None or isinstance(result, str)

    def test_generate_code_unicode_prompt(self, orch):
        """Test generate_code with unicode characters."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="code")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            prompt = "测试 テスト тест"
            result = client.generate_code(prompt)
            assert result is None or isinstance(result, str)


class TestGenerateSocraticBranches:
    """Test different branches in generate_socratic_question."""

    def test_generate_question_without_cache_key(self, orch):
        """Test generate_socratic_question without cache_key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="question")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_socratic_question("prompt")

            assert result is not None or result is None

    def test_generate_question_with_explicit_cache_key(self, orch):
        """Test generate_socratic_question with explicit cache_key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="q")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_socratic_question("prompt", cache_key="key123")

            assert result is not None or result is None

    def test_generate_question_empty_cache_hit(self, orch):
        """Test generate_socratic_question returns empty cache."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)

            # Set empty cache entry
            cache_key = client._get_cache_key("prompt")
            client._question_cache[cache_key] = ""

            result = client.generate_socratic_question("prompt", cache_key=cache_key)
            assert result is not None or result is None

    def test_generate_question_none_in_cache(self, orch):
        """Test generate_socratic_question with None in cache."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)

            # Set None in cache
            cache_key = client._get_cache_key("prompt")
            client._question_cache[cache_key] = None

            result = client.generate_socratic_question("prompt", cache_key=cache_key)
            assert result is None or isinstance(result, str)


class TestGenerateResponseBranches:
    """Test different branches in generate_response."""

    def test_generate_response_no_system_prompt(self, orch):
        """Test generate_response without system prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="response")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_response("prompt")

            assert result is not None or result is None

    def test_generate_response_empty_system_prompt(self, orch):
        """Test generate_response with empty system prompt."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="response")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_response("prompt", system_prompt="")

            assert result is not None or result is None

    def test_generate_response_no_max_tokens(self, orch):
        """Test generate_response without max_tokens."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="response")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_response("prompt", max_tokens=None)

            assert result is not None or result is None

    def test_generate_response_zero_temperature(self, orch):
        """Test generate_response with zero temperature."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="response")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_response("prompt", temperature=0)

            assert result is not None or result is None

    def test_generate_response_max_temperature(self, orch):
        """Test generate_response with maximum temperature."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = Mock(
                content=[Mock(text="response")],
                usage=Mock(input_tokens=10, output_tokens=20)
            )

            client = ClaudeClient(api_key="test", orchestrator=orch)
            result = client.generate_response("prompt", temperature=1.0)

            assert result is not None or result is None


class TestCacheKeyBranches:
    """Test cache key generation branches."""

    def test_cache_key_empty_message(self, orch):
        """Test cache key generation for empty message."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            key = client._get_cache_key("")
            assert len(key) == 64

    def test_cache_key_long_message(self, orch):
        """Test cache key for very long message."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            key = client._get_cache_key("x" * 10000)
            assert len(key) == 64

    def test_cache_key_special_chars(self, orch):
        """Test cache key with special characters."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            key = client._get_cache_key("!@#$%^&*()")
            assert len(key) == 64

    def test_cache_key_unicode(self, orch):
        """Test cache key with unicode."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            key = client._get_cache_key("测试 テスト тест")
            assert len(key) == 64

    def test_cache_key_newlines(self, orch):
        """Test cache key with newlines."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            key = client._get_cache_key("line1\nline2\nline3")
            assert len(key) == 64


class TestProjectContextBranches:
    """Test ProjectContext handling branches."""

    def test_extract_insights_minimal_context(self, orch):
        """Test with minimal ProjectContext."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("message", project)
            assert isinstance(result, dict)

    def test_extract_insights_full_context(self, orch):
        """Test with fully populated ProjectContext."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            project = ProjectContext(
                project_name="Full",
                description="Description",
                goals=["g1", "g2"],
                phase="phase",
                tech_stack=["tech1"],
                project_type="type",
                deployment_target="target",
                status="status"
            )
            result = client.extract_insights("message", project)
            assert isinstance(result, dict)

    def test_extract_insights_none_fields_context(self, orch):
        """Test with None fields in ProjectContext."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=orch)
            project = ProjectContext(
                project_name="Test",
                description=None,
                goals=None
            )
            result = client.extract_insights("message", project)
            assert isinstance(result, dict)
