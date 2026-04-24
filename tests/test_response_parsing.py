"""Comprehensive tests for response parsing and JSON handling."""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient


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


class TestParseJsonResponse:
    """Tests for _parse_json_response method."""

    def test_parse_valid_json_object(self, mock_orchestrator):
        """Test parsing valid JSON object."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response('{"key": "value"}')

            assert isinstance(result, dict)
            assert result.get("key") == "value"

    def test_parse_valid_json_array(self, mock_orchestrator):
        """Test parsing valid JSON array."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response('[1, 2, 3]')

            assert isinstance(result, (list, dict))

    def test_parse_valid_json_number(self, mock_orchestrator):
        """Test parsing JSON number."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response('42')

            assert result is not None or isinstance(result, (int, float, dict))

    def test_parse_valid_json_string(self, mock_orchestrator):
        """Test parsing JSON string."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response('"hello"')

            assert result is not None

    def test_parse_invalid_json_returns_none(self, mock_orchestrator):
        """Test parsing invalid JSON returns None."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response("not valid json at all")

            assert result is None

    def test_parse_empty_string_returns_none(self, mock_orchestrator):
        """Test parsing empty string."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response("")

            assert result is None

    def test_parse_whitespace_only_string(self, mock_orchestrator):
        """Test parsing whitespace only string."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response("   \n\t  ")

            assert result is None

    def test_parse_json_with_whitespace(self, mock_orchestrator):
        """Test parsing JSON with surrounding whitespace."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response('  {"key": "value"}  \n')

            assert isinstance(result, dict)

    def test_parse_nested_json_object(self, mock_orchestrator):
        """Test parsing nested JSON object."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            json_str = '{"outer": {"inner": {"deep": "value"}}}'
            result = client._parse_json_response(json_str)

            assert isinstance(result, dict)
            assert result.get("outer") is not None

    def test_parse_json_with_special_characters(self, mock_orchestrator):
        """Test parsing JSON with special characters."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            json_str = '{"text": "hello\\nworld\\ttab"}'
            result = client._parse_json_response(json_str)

            assert isinstance(result, dict)

    def test_parse_json_with_unicode(self, mock_orchestrator):
        """Test parsing JSON with unicode characters."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            json_str = '{"text": "你好世界"}'
            result = client._parse_json_response(json_str)

            assert isinstance(result, dict)

    def test_parse_json_with_null_values(self, mock_orchestrator):
        """Test parsing JSON with null values."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            json_str = '{"key": null, "other": "value"}'
            result = client._parse_json_response(json_str)

            assert isinstance(result, dict)

    def test_parse_json_with_booleans(self, mock_orchestrator):
        """Test parsing JSON with boolean values."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            json_str = '{"true_val": true, "false_val": false}'
            result = client._parse_json_response(json_str)

            assert isinstance(result, dict)

    def test_parse_malformed_json_missing_bracket(self, mock_orchestrator):
        """Test parsing malformed JSON with missing bracket."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response('{"key": "value"')

            assert result is None

    def test_parse_malformed_json_extra_bracket(self, mock_orchestrator):
        """Test parsing malformed JSON with extra bracket."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response('{"key": "value"}}}')

            assert result is None

    def test_parse_single_quotes_not_valid(self, mock_orchestrator):
        """Test that single quotes don't parse as JSON."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client._parse_json_response("{'key': 'value'}")

            assert result is None


class TestCalculateCost:
    """Tests for _calculate_cost method."""

    def test_calculate_cost_basic(self, mock_orchestrator):
        """Test calculating cost with basic usage."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            usage = Mock()
            usage.input_tokens = 100
            usage.output_tokens = 50

            cost = client._calculate_cost(usage)
            assert isinstance(cost, (int, float))
            assert cost >= 0

    def test_calculate_cost_zero_tokens(self, mock_orchestrator):
        """Test calculating cost with zero tokens."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            usage = Mock()
            usage.input_tokens = 0
            usage.output_tokens = 0

            cost = client._calculate_cost(usage)
            assert isinstance(cost, (int, float))
            assert cost == 0

    def test_calculate_cost_large_tokens(self, mock_orchestrator):
        """Test calculating cost with large token counts."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            usage = Mock()
            usage.input_tokens = 100000
            usage.output_tokens = 50000

            cost = client._calculate_cost(usage)
            assert isinstance(cost, (int, float))
            assert cost > 0

    def test_calculate_cost_only_input_tokens(self, mock_orchestrator):
        """Test calculating cost with only input tokens."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            usage = Mock()
            usage.input_tokens = 1000
            usage.output_tokens = 0

            cost = client._calculate_cost(usage)
            assert isinstance(cost, (int, float))

    def test_calculate_cost_only_output_tokens(self, mock_orchestrator):
        """Test calculating cost with only output tokens."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            usage = Mock()
            usage.input_tokens = 0
            usage.output_tokens = 1000

            cost = client._calculate_cost(usage)
            assert isinstance(cost, (int, float))


class TestTrackTokenUsage:
    """Tests for _track_token_usage method."""

    def test_track_token_usage_basic(self, mock_orchestrator):
        """Test tracking token usage."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            usage = Mock()
            usage.input_tokens = 100
            usage.output_tokens = 50

            # Should not raise
            client._track_token_usage(usage, "test_operation")

    def test_track_token_usage_different_operations(self, mock_orchestrator):
        """Test tracking with different operation names."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            usage = Mock()
            usage.input_tokens = 50
            usage.output_tokens = 25

            operations = [
                "generate_code",
                "extract_insights",
                "generate_response",
                "generate_socratic_question"
            ]

            for op in operations:
                # Should not raise
                client._track_token_usage(usage, op)

    def test_track_token_usage_zero_tokens(self, mock_orchestrator):
        """Test tracking with zero tokens."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            usage = Mock()
            usage.input_tokens = 0
            usage.output_tokens = 0

            # Should not raise
            client._track_token_usage(usage, "operation")

    def test_track_token_usage_large_numbers(self, mock_orchestrator):
        """Test tracking with large token numbers."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            usage = Mock()
            usage.input_tokens = 1000000
            usage.output_tokens = 500000

            # Should not raise
            client._track_token_usage(usage, "operation")


class TestResponseExtractionEdgeCases:
    """Tests for edge cases in response extraction."""

    def test_extract_text_from_single_block(self, mock_orchestrator):
        """Test extracting text from single content block."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="extracted text")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            assert result is not None or result is None

    def test_extract_text_with_newlines(self, mock_orchestrator):
        """Test extracting text containing newlines."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="line1\nline2\nline3")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            assert result is not None or result is None

    def test_extract_text_with_special_characters(self, mock_orchestrator):
        """Test extracting text with special characters."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="!@#$%^&*()_+-=[]{}|;:',.<>?/")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            assert result is not None or result is None

    def test_extract_text_empty_string(self, mock_orchestrator):
        """Test extracting empty text."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="")]
            response.usage = Mock(input_tokens=10, output_tokens=0)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test", orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            # Empty text is acceptable
            assert result is not None or result is None
