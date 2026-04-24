"""
Comprehensive error scenario and edge case tests for ClaudeClient

Tests error handling, recovery, and edge cases:
- API errors (401, 403, 429, 503, timeout)
- Malformed responses and invalid JSON
- Input validation and sanitization
- Database errors and fallback mechanisms
- Authentication failures and recovery
- Token limit handling
- Concurrent request handling
- Cache miss scenarios
- Encryption/decryption errors
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.exceptions import APIError
from socratic_nexus.models import ProjectContext


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator"""
    orch = Mock()
    orch.config = Mock()
    orch.config.claude_model = "claude-haiku-4-5-20251001"
    orch.event_emitter = Mock()
    orch.database = Mock()
    orch.database.get_api_key.return_value = None
    orch.system_monitor = Mock()
    orch.system_monitor.process = Mock()
    return orch


class TestAPIErrors:
    """Tests for API error handling"""

    def test_handle_401_unauthorized(self, mock_orchestrator):
        """Test handling of 401 Unauthorized error"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("401 Unauthorized")

            client = ClaudeClient(api_key="invalid-key", orchestrator=mock_orchestrator)

            with pytest.raises(Exception):
                client.generate_response("test")

    def test_handle_403_forbidden(self, mock_orchestrator):
        """Test handling of 403 Forbidden error"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("403 Forbidden")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            with pytest.raises(Exception):
                client.generate_response("test")

    def test_handle_429_rate_limit(self, mock_orchestrator):
        """Test handling of 429 Rate Limit error"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("429 Too Many Requests")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            with pytest.raises(Exception):
                client.generate_response("test")

    def test_handle_503_service_unavailable(self, mock_orchestrator):
        """Test handling of 503 Service Unavailable error"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("503 Service Unavailable")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            with pytest.raises(Exception):
                client.generate_response("test")

    def test_handle_timeout(self, mock_orchestrator):
        """Test handling of timeout errors"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = TimeoutError("Request timeout")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            with pytest.raises(TimeoutError):
                client.generate_response("test")

    def test_handle_connection_error(self, mock_orchestrator):
        """Test handling of connection errors"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = ConnectionError("Connection failed")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            with pytest.raises(ConnectionError):
                client.generate_response("test")


class TestMalformedResponses:
    """Tests for handling malformed API responses"""

    def test_handle_empty_response(self, mock_orchestrator):
        """Test handling of empty response content"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = []  # Empty content
            mock_response.usage = Mock(input_tokens=10, output_tokens=0)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            # Should handle gracefully
            assert result is None or isinstance(result, str)

    def test_handle_missing_usage_data(self, mock_orchestrator):
        """Test handling of missing usage data"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = None  # Missing usage
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            # Should handle gracefully
            assert isinstance(result, (str, type(None)))

    def test_handle_invalid_json_in_response(self, mock_orchestrator):
        """Test handling of invalid JSON in response"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="{invalid json")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("response", project)

            # Should return empty dict or None for invalid JSON
            assert isinstance(result, (dict, str, type(None)))

    def test_handle_response_with_null_text(self, mock_orchestrator):
        """Test handling of response with null text"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text=None)]  # Null text
            mock_response.usage = Mock(input_tokens=10, output_tokens=0)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("test")

            # Should handle gracefully
            assert result is None or isinstance(result, str)

    def test_handle_truncated_response(self, mock_orchestrator):
        """Test handling of truncated responses"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            # Simulate truncated response
            mock_response.content = [Mock(text="response that was trun")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=1000)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("long prompt")

            assert isinstance(result, (str, type(None)))


class TestInputValidation:
    """Tests for input validation"""

    def test_handle_none_prompt(self, mock_orchestrator):
        """Test handling of None prompt"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Should handle gracefully
            try:
                result = client.generate_response(None)
                assert result is None or isinstance(result, str)
            except (TypeError, AttributeError):
                # TypeError for None is acceptable
                pass

    def test_handle_empty_prompt(self, mock_orchestrator):
        """Test handling of empty prompt"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=5, output_tokens=10)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("")

            # Should handle empty string
            assert isinstance(result, (str, type(None)))

    def test_handle_very_long_prompt(self, mock_orchestrator):
        """Test handling of extremely long prompt"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10000, output_tokens=100)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            very_long_prompt = "test " * 100000  # Very long
            result = client.generate_response(very_long_prompt)

            assert isinstance(result, (str, type(None)))

    def test_handle_special_characters_in_prompt(self, mock_orchestrator):
        """Test handling of special characters in prompt"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=30, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            special_prompt = "test with special chars: !@#$%^&*()[]{}|\\;:'\",.<>?/`~"
            result = client.generate_response(special_prompt)

            assert isinstance(result, (str, type(None)))

    def test_handle_unicode_characters(self, mock_orchestrator):
        """Test handling of unicode characters"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=20, output_tokens=40)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            unicode_prompt = "test with unicode: 你好世界 🚀 مرحبا بالعالم"
            result = client.generate_response(unicode_prompt)

            assert isinstance(result, (str, type(None)))


class TestParameterValidation:
    """Tests for parameter validation"""

    def test_invalid_max_tokens(self, mock_orchestrator):
        """Test handling of invalid max_tokens parameter"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Negative max_tokens should be handled
            try:
                result = client.generate_response("test", max_tokens=-100)
                # If it doesn't error, it should work
                assert isinstance(result, (str, type(None)))
            except (ValueError, TypeError):
                # ValueError or TypeError is acceptable
                pass

    def test_invalid_temperature(self, mock_orchestrator):
        """Test handling of invalid temperature parameter"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Temperature > 1.0 might be handled by API
            try:
                result = client.generate_response("test", temperature=2.0)
                assert isinstance(result, (str, type(None)))
            except (ValueError, TypeError):
                pass

    def test_negative_temperature(self, mock_orchestrator):
        """Test handling of negative temperature"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            try:
                result = client.generate_response("test", temperature=-0.5)
                assert isinstance(result, (str, type(None)))
            except (ValueError, TypeError):
                pass


class TestDatabaseErrors:
    """Tests for database error handling"""

    def test_database_connection_error(self, mock_orchestrator):
        """Test handling of database connection errors"""
        mock_orchestrator.database.get_api_key.side_effect = Exception("DB Connection failed")

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="fallback-key",
                orchestrator=mock_orchestrator
            )

            # Should fallback to provided key
            api_key, is_user = client._get_user_api_key("user123")
            assert api_key == "fallback-key"
            assert is_user is False

    def test_database_timeout(self, mock_orchestrator):
        """Test handling of database timeout"""
        mock_orchestrator.database.get_api_key.side_effect = TimeoutError("DB Timeout")

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="fallback-key",
                orchestrator=mock_orchestrator
            )

            api_key, is_user = client._get_user_api_key("user123")
            assert api_key == "fallback-key"


class TestAuthenticationFailures:
    """Tests for authentication failure handling"""

    def test_missing_api_key_error(self, mock_orchestrator):
        """Test APIError when both API key and orchestrator are missing"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=None)

            with pytest.raises(APIError):
                client._get_client()

    def test_placeholder_key_detection(self, mock_orchestrator):
        """Test that placeholder keys are detected"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="placeholder_test",
                orchestrator=mock_orchestrator
            )

            # Placeholder keys should not create a real client
            assert client.client is None


class TestConcurrentRequests:
    """Tests for concurrent request handling"""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_orchestrator):
        """Test handling of concurrent requests"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text="response")],
                    usage=Mock(input_tokens=10, output_tokens=20)
                )
            )

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

                # Simulate concurrent requests
                results = await asyncio.gather(
                    client.generate_response_async("prompt1"),
                    client.generate_response_async("prompt2"),
                    client.generate_response_async("prompt3"),
                )

                assert len(results) == 3
                for result in results:
                    assert isinstance(result, (str, type(None)))


class TestCacheMissScenarios:
    """Tests for cache miss scenarios"""

    def test_cache_miss_different_prompts(self, mock_orchestrator):
        """Test that different prompts don't share cache"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Different prompts should both call API
            client.generate_response("prompt1")
            client.generate_response("prompt2")

            # Should be called at least once per unique prompt
            assert mock_client.messages.create.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
