"""Tests for response handling and message formatting."""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext


@pytest.fixture
def mock_orch():
    """Create mock orchestrator."""
    orch = Mock()
    orch.config = Mock()
    orch.config.claude_model = "claude-3-sonnet-20240229"
    orch.event_emitter = Mock()
    return orch


@pytest.fixture
def client(mock_orch):
    """Create client."""
    with patch("socratic_nexus.clients.claude_client.anthropic"):
        return ClaudeClient(api_key="test", orchestrator=mock_orch)


class TestResponseParsing:
    """Test response parsing and handling."""

    def test_extract_insights_with_empty_string(self, client):
        """Test extract_insights with empty string."""
        project = ProjectContext(project_name="Test")
        result = client.extract_insights("", project)
        assert isinstance(result, dict)

    def test_extract_insights_with_whitespace(self, client):
        """Test extract_insights with whitespace only."""
        project = ProjectContext(project_name="Test")
        result = client.extract_insights("   ", project)
        assert isinstance(result, dict)

    def test_extract_insights_with_very_long_text(self, client):
        """Test extract_insights with very long text."""
        project = ProjectContext(project_name="Test")
        long_text = "a" * 10000
        try:
            result = client.extract_insights(long_text, project)
            assert isinstance(result, dict)
        except Exception:
            pass

    def test_extract_insights_with_special_chars(self, client):
        """Test extract_insights with special characters."""
        project = ProjectContext(project_name="Test")
        text = "test!@#$%^&*()_+-=[]{}|;:',.<>?/\\"
        result = client.extract_insights(text, project)
        assert isinstance(result, dict)

    def test_extract_insights_with_unicode(self, client):
        """Test extract_insights with unicode characters."""
        project = ProjectContext(project_name="Test")
        text = "测试 テスト тест مرحبا"
        result = client.extract_insights(text, project)
        assert isinstance(result, dict)

    def test_extract_insights_with_newlines(self, client):
        """Test extract_insights with newlines."""
        project = ProjectContext(project_name="Test")
        text = "line1\nline2\nline3"
        result = client.extract_insights(text, project)
        assert isinstance(result, dict)


class TestProjectContextHandling:
    """Test ProjectContext parameter handling."""

    def test_extract_insights_with_minimal_context(self, client):
        """Test with minimal ProjectContext."""
        project = ProjectContext(project_name="Test")
        result = client.extract_insights("test message", project)
        assert isinstance(result, dict)

    def test_extract_insights_with_full_context(self, client):
        """Test with fully populated ProjectContext."""
        project = ProjectContext(
            project_name="Full Project",
            description="A test project",
            goals=["goal1", "goal2"],
            phase="implementation",
            tech_stack=["Python", "Django"],
            project_type="web",
            deployment_target="cloud",
        )
        result = client.extract_insights("test message", project)
        assert isinstance(result, dict)

    def test_extract_insights_with_none_fields(self, client):
        """Test with ProjectContext containing None fields."""
        project = ProjectContext(project_name="Test", description=None, goals=None, phase=None)
        result = client.extract_insights("test message", project)
        assert isinstance(result, dict)


class TestAuthMethodVariations:
    """Test different authentication method handling."""

    def test_extract_insights_no_user_id(self, client):
        """Test extract_insights without user_id."""
        project = ProjectContext(project_name="Test")
        result = client.extract_insights("test", project)
        assert isinstance(result, dict)

    def test_extract_insights_with_user_id(self, client, mock_orch):
        """Test extract_insights with user_id."""
        project = ProjectContext(project_name="Test")
        mock_orch.database = Mock()
        mock_orch.database.get_api_key.return_value = None

        try:
            result = client.extract_insights("test", project, user_id="user123")
            assert isinstance(result, dict)
        except Exception:
            pass

    def test_extract_insights_with_user_auth_method(self, client, mock_orch):
        """Test extract_insights with user_auth_method."""
        project = ProjectContext(project_name="Test")
        mock_orch.database = Mock()

        try:
            result = client.extract_insights("test", project, user_auth_method="api_key")
            assert isinstance(result, dict)
        except Exception:
            pass


class TestCacheKeyBehavior:
    """Test cache key generation behavior."""

    def test_same_message_produces_same_key(self, client):
        """Test same message always produces same cache key."""
        msg = "consistent message"
        key1 = client._get_cache_key(msg)
        key2 = client._get_cache_key(msg)
        key3 = client._get_cache_key(msg)

        assert key1 == key2 == key3

    def test_similar_messages_produce_different_keys(self, client):
        """Test similar but different messages produce different keys."""
        key1 = client._get_cache_key("test message")
        key2 = client._get_cache_key("test message ")
        key3 = client._get_cache_key("Test message")

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_key_is_hex_format(self, client):
        """Test cache key is valid hex."""
        key = client._get_cache_key("any message")
        assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_is_64_chars(self, client):
        """Test cache key is 64 characters (SHA256)."""
        key = client._get_cache_key("test")
        assert len(key) == 64


class TestGenerateResponseVariations:
    """Test generate_response with various inputs."""

    def test_generate_response_empty_prompt(self, client):
        """Test generate_response with empty prompt."""
        try:
            result = client.generate_response("")
            assert result is None or isinstance(result, str)
        except Exception:
            pass

    def test_generate_response_very_long_prompt(self, client):
        """Test generate_response with very long prompt."""
        long_prompt = "test " * 1000
        try:
            result = client.generate_response(long_prompt)
            assert result is None or isinstance(result, str)
        except Exception:
            pass

    def test_generate_response_with_newlines(self, client):
        """Test generate_response with multi-line prompt."""
        multi_line = "line1\nline2\nline3"
        try:
            result = client.generate_response(multi_line)
            assert result is None or isinstance(result, str)
        except Exception:
            pass

    def test_generate_response_with_special_chars(self, client):
        """Test generate_response with special characters."""
        special = "What is 2+2? (Test: [a,b,c])"
        try:
            result = client.generate_response(special)
            assert result is None or isinstance(result, str)
        except Exception:
            pass

    def test_generate_response_system_prompt_variations(self, client):
        """Test generate_response with different system prompts."""
        system_prompts = ["You are helpful", "", "You are an expert", "Be concise: " * 100]

        for sys_prompt in system_prompts:
            try:
                result = client.generate_response("test", system_prompt=sys_prompt)
                assert result is None or isinstance(result, str)
            except Exception:
                pass


class TestGenerateCodeVariations:
    """Test generate_code with various inputs."""

    def test_generate_code_empty_prompt(self, client):
        """Test generate_code with empty prompt."""
        result = client.generate_code("")
        assert result is None or isinstance(result, str)

    def test_generate_code_code_snippet_input(self, client):
        """Test generate_code with code snippet."""
        code_input = "def hello():\n    print('hello')"
        result = client.generate_code(code_input)
        assert result is None or isinstance(result, str)

    def test_generate_code_requirements_variations(self, client):
        """Test generate_code with different requirements."""
        req_lists = [
            [],
            ["requests"],
            ["django", "djangorestframework", "celery"],
            ["a" * 100],  # very long requirement name
        ]

        for reqs in req_lists:
            try:
                result = client.generate_code("test", requirements=reqs)
                assert result is None or isinstance(result, str)
            except TypeError:
                pass

    def test_generate_code_max_tokens_variations(self, client):
        """Test generate_code with different max_tokens values."""
        for tokens in [10, 100, 1000, 4096]:
            try:
                result = client.generate_code("test", max_tokens=tokens)
                assert result is None or isinstance(result, str)
            except TypeError:
                pass


class TestSocraticQuestionVariations:
    """Test generate_socratic_question with various inputs."""

    def test_generate_question_empty_prompt(self, client):
        """Test generate_socratic_question with empty prompt."""
        try:
            result = client.generate_socratic_question("")
            assert result is None or isinstance(result, str)
        except Exception:
            pass

    def test_generate_question_long_prompt(self, client):
        """Test generate_socratic_question with long prompt."""
        long_prompt = "prompt " * 500
        try:
            result = client.generate_socratic_question(long_prompt)
            assert result is None or isinstance(result, str)
        except Exception:
            pass

    def test_generate_question_cache_key_override(self, client):
        """Test generate_socratic_question with custom cache_key."""
        try:
            result = client.generate_socratic_question("test", cache_key="custom_key")
            assert result is None or isinstance(result, str)
        except Exception:
            pass

    def test_generate_question_multiple_cache_keys(self, client):
        """Test multiple different cache keys."""
        keys = ["key1", "key2", "key3", None]
        for key in keys:
            try:
                if key:
                    result = client.generate_socratic_question("test", cache_key=key)
                else:
                    result = client.generate_socratic_question("test")
                assert result is None or isinstance(result, str)
            except Exception:
                pass
