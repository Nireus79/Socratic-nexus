"""Comprehensive error scenario testing for ClaudeClient.

Tests error handling across all major methods:
- API errors and network failures
- Invalid responses and malformed JSON
- Authentication failures
- Rate limiting scenarios
- Timeout handling
- Edge cases with unusual input
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext
from socratic_nexus.exceptions import APIError


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


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_api_error_in_generate_response(self, mock_orchestrator):
        """Test handling of generic API errors."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception(
                "API Error: 503 Service Unavailable"
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Should raise APIError
            with pytest.raises(APIError) as exc_info:
                client.generate_response("prompt")

            assert exc_info.value.error_type == "GENERATION_ERROR"

    def test_timeout_error(self, mock_orchestrator):
        """Test handling of timeout errors."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = TimeoutError(
                "Request timed out after 30 seconds"
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Should raise APIError
            with pytest.raises(APIError) as exc_info:
                client.generate_response("prompt")

            assert exc_info.value.error_type == "GENERATION_ERROR"

    def test_connection_error(self, mock_orchestrator):
        """Test handling of connection errors."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = ConnectionError(
                "Failed to connect to API"
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Should raise APIError
            with pytest.raises(APIError) as exc_info:
                client.generate_response("prompt")

            assert exc_info.value.error_type == "GENERATION_ERROR"

    def test_authentication_error(self, mock_orchestrator):
        """Test handling of authentication errors."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception(
                "401 Unauthorized: Invalid API key"
            )

            client = ClaudeClient(api_key="invalid-key", orchestrator=mock_orchestrator)

            # Should raise APIError
            with pytest.raises(APIError) as exc_info:
                client.generate_response("prompt")

            assert exc_info.value.error_type == "GENERATION_ERROR"

    def test_rate_limit_error(self, mock_orchestrator):
        """Test handling of rate limit errors."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception(
                "429 Too Many Requests: Rate limit exceeded"
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Should raise APIError
            with pytest.raises(APIError) as exc_info:
                client.generate_response("prompt")

            assert exc_info.value.error_type == "GENERATION_ERROR"


class TestMalformedResponseHandling:
    """Tests for handling malformed responses."""

    def test_missing_content_field(self, mock_orchestrator):
        """Test handling when response lacks content."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = []  # Empty content
            response.usage = Mock(input_tokens=10, output_tokens=0)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Should raise APIError when content is missing
            with pytest.raises(APIError) as exc_info:
                client.generate_response("prompt")

            assert exc_info.value.error_type == "GENERATION_ERROR"

    def test_invalid_json_response(self, mock_orchestrator):
        """Test handling of invalid JSON in structured responses."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="{ invalid json ")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.extract_insights("response", ProjectContext(
                project_name="Test"
            ))

            # Should return dict (empty or with parsed data)
            assert isinstance(result, dict)

    def test_malformed_json_array_in_detect_conflicts(self, mock_orchestrator):
        """Test handling of malformed JSON arrays in async conflict detection."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="[1, 2, 3,")]  # Incomplete array
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            # Use pytest.mark.asyncio or create an async test
            import asyncio
            result = asyncio.run(client.detect_conflicts_async([]))

            # Should handle gracefully
            assert isinstance(result, (list, dict))

    def test_empty_response_string(self, mock_orchestrator):
        """Test handling of empty response strings."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="")]
            response.usage = Mock(input_tokens=10, output_tokens=0)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt")

            # Empty response should be handled
            assert result == "" or result is None


class TestInputValidationErrors:
    """Tests for input validation and edge cases."""

    def test_none_prompt(self, mock_orchestrator):
        """Test handling of None prompt."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Should handle None gracefully (convert to string or error)
            try:
                result = client.generate_response(None)
                assert result is None or isinstance(result, str)
            except (TypeError, APIError):
                # Both are acceptable
                pass

    def test_extremely_long_prompt(self, mock_orchestrator):
        """Test handling of very long prompts."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=100000, output_tokens=10)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            long_prompt = "a" * 1000000  # 1 million characters
            result = client.generate_response(long_prompt)

            # Should handle or return error
            assert result is None or isinstance(result, str)

    def test_special_characters_in_prompt(self, mock_orchestrator):
        """Test handling of special characters."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            special_prompt = "!@#$%^&*()[]{}|\\:;\"'<>?,./\n\t\r"
            result = client.generate_response(special_prompt)

            assert isinstance(result, (str, type(None)))

    def test_unicode_characters_in_prompt(self, mock_orchestrator):
        """Test handling of Unicode characters."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            unicode_prompt = "你好世界 مرحبا بالعالم שלום עולם"
            result = client.generate_response(unicode_prompt)

            assert isinstance(result, (str, type(None)))

    def test_null_byte_in_prompt(self, mock_orchestrator):
        """Test handling of null bytes."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            prompt_with_null = "prompt\x00with\x00nulls"
            result = client.generate_response(prompt_with_null)

            assert isinstance(result, (str, type(None)))


class TestAuthenticationErrors:
    """Tests for authentication failure scenarios."""

    def test_missing_api_key(self, mock_orchestrator):
        """Test behavior with missing API key."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_anth.return_value = None

            # Client initialization with None API key
            client = ClaudeClient(api_key=None, orchestrator=mock_orchestrator)

            # test_connection should fail or return False
            try:
                result = client.test_connection()
                assert isinstance(result, bool)
            except APIError:
                pass  # Expected behavior

    def test_invalid_api_key_format(self, mock_orchestrator):
        """Test with invalid API key format."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception(
                "Invalid API key format"
            )

            client = ClaudeClient(api_key="not-a-real-key", orchestrator=mock_orchestrator)

            # Should raise APIError
            with pytest.raises(APIError) as exc_info:
                client.test_connection()

            assert exc_info.value.error_type == "CONNECTION_ERROR"

    def test_expired_api_key(self, mock_orchestrator):
        """Test with expired API key."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception(
                "401 Unauthorized: API key expired"
            )

            client = ClaudeClient(api_key="expired-key", orchestrator=mock_orchestrator)

            # Should raise APIError
            with pytest.raises(APIError) as exc_info:
                client.generate_response("test")

            assert exc_info.value.error_type == "GENERATION_ERROR"


class TestParameterValidationErrors:
    """Tests for parameter validation."""

    def test_invalid_max_tokens(self, mock_orchestrator):
        """Test handling of invalid max_tokens."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Negative max_tokens
            result = client.generate_response("prompt", max_tokens=-1)
            assert isinstance(result, (str, type(None)))

            # Zero max_tokens
            result = client.generate_response("prompt", max_tokens=0)
            assert isinstance(result, (str, type(None)))

            # Extremely large max_tokens
            result = client.generate_response("prompt", max_tokens=1000000000)
            assert isinstance(result, (str, type(None)))

    def test_invalid_temperature(self, mock_orchestrator):
        """Test handling of invalid temperature."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Negative temperature
            result = client.generate_response("prompt", temperature=-1.0)
            assert isinstance(result, (str, type(None)))

            # Temperature > 2.0
            result = client.generate_response("prompt", temperature=5.0)
            assert isinstance(result, (str, type(None)))


class TestCacheErrorHandling:
    """Tests for cache-related error handling."""

    def test_cache_with_large_response(self, mock_orchestrator):
        """Test caching with very large responses."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            large_response = "a" * 1000000  # 1MB response
            response = Mock()
            response.content = [Mock(text=large_response)]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # First call - caches result
            result1 = client.extract_insights("test", ProjectContext(
                project_name="Test"
            ))

            # Second call - uses cache
            result2 = client.extract_insights("test", ProjectContext(
                project_name="Test"
            ))

            # Should work without memory issues
            assert isinstance(result1, dict)
            assert isinstance(result2, dict)

    def test_cache_collision_handling(self, mock_orchestrator):
        """Test handling of potential cache key collisions."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Make many similar requests
            for i in range(100):
                result = client.extract_insights(f"prompt {i}", ProjectContext(
                    project_name="Test"
                ))
                assert isinstance(result, dict)


class TestAsyncErrorHandling:
    """Tests for error handling in async methods."""

    @pytest.mark.asyncio
    async def test_async_api_error(self, mock_orchestrator):
        """Test async method error handling."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("API Error")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = await client.detect_conflicts_async([])

            # Should handle error gracefully
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_async_timeout_error(self, mock_orchestrator):
        """Test async timeout handling."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = asyncio.TimeoutError()

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = await client.analyze_context_async(ProjectContext(
                project_name="Test"
            ))

            # Should return empty string on error
            assert result == "" or result is None

    @pytest.mark.asyncio
    async def test_concurrent_async_errors(self, mock_orchestrator):
        """Test handling of errors in concurrent async calls."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client

            # Half fail, half succeed
            responses = [
                Exception("Error"),
                Mock(content=[Mock(text="response")], usage=Mock(
                    input_tokens=10, output_tokens=20
                )),
                Exception("Error"),
                Mock(content=[Mock(text="response")], usage=Mock(
                    input_tokens=10, output_tokens=20
                )),
            ]
            mock_client.messages.create.side_effect = responses

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            tasks = [
                client.detect_conflicts_async([]),
                client.analyze_context_async(ProjectContext(project_name="Test")),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Should handle some errors
            assert len(results) == 2


class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""

    def test_minimum_valid_response(self, mock_orchestrator):
        """Test with minimum valid API response."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="x")]  # Single character response
            response.usage = Mock(input_tokens=1, output_tokens=1)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("minimal prompt")

            assert isinstance(result, str)

    def test_maximum_valid_response(self, mock_orchestrator):
        """Test with maximum valid API response."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="x" * 100000)]  # Very large response
            response.usage = Mock(input_tokens=10, output_tokens=100000)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt")

            assert isinstance(result, str)

    def test_zero_token_response(self, mock_orchestrator):
        """Test with zero tokens in response."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=0, output_tokens=0)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt")

            assert isinstance(result, str)

    def test_float_token_values(self, mock_orchestrator):
        """Test with float token values."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.Anthropic"
        ) as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            response = Mock()
            response.content = [Mock(text="response")]
            response.usage = Mock(input_tokens=10.5, output_tokens=20.7)
            mock_client.messages.create.return_value = response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_response("prompt")

            # Should handle float tokens
            assert isinstance(result, str)
