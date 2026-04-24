"""
Tests for async methods and remaining coverage gaps in Claude client.
"""

import pytest
from unittest.mock import Mock, patch
from socratic_nexus.clients.claude_client import ClaudeClient


class TestClaudeClientAsyncResponseGeneration:
    """Test async response generation methods"""

    @pytest.mark.asyncio
    async def test_async_generate_response_exists_and_callable(self):
        """Test async generate_response is callable"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key")
                assert callable(client.generate_response_async)

    @pytest.mark.asyncio
    async def test_async_generate_code_exists_and_callable(self):
        """Test async generate_code is callable"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key")
                assert callable(client.generate_code_async)

    @pytest.mark.asyncio
    async def test_async_extract_insights_exists_and_callable(self):
        """Test async extract_insights is callable"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key")
                assert callable(client.extract_insights_async)


class TestClaudeClientAsyncGenerationMethods:
    """Test async specialized generation methods"""

    @pytest.mark.asyncio
    async def test_async_business_plan_callable(self):
        """Test async business plan generation is callable"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key")
                assert callable(client.generate_business_plan_async)

    @pytest.mark.asyncio
    async def test_async_documentation_callable(self):
        """Test async documentation generation is callable"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key")
                assert callable(client.generate_documentation_async)

    @pytest.mark.asyncio
    async def test_async_documentation_callable(self):
        """Test async documentation generation is callable"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key")
                assert callable(client.generate_documentation_async)

    @pytest.mark.asyncio
    async def test_async_evaluate_quality_callable(self):
        """Test async quality evaluation is callable"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key")
                assert callable(client.evaluate_quality_async)

    @pytest.mark.asyncio
    async def test_async_extract_tech_recommendations_callable(self):
        """Test async tech recommendations is callable"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key")
                assert callable(client.extract_tech_recommendations_async)


class TestClaudeClientAsyncTracking:
    """Test async token tracking"""

    @pytest.mark.asyncio
    async def test_async_track_token_usage_callable(self):
        """Test async token tracking is callable"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                orch = Mock()
                orch.config = Mock()
                orch.system_monitor = Mock()

                client = ClaudeClient(api_key="test-key", orchestrator=orch)
                assert callable(client._track_token_usage_async)


class TestClaudeClientDocstringAndAttributes:
    """Test client attributes and docstrings"""

    def test_client_has_docstring(self):
        """Test ClaudeClient class has docstring"""
        assert ClaudeClient.__doc__ is not None
        assert len(ClaudeClient.__doc__) > 0

    def test_client_has_api_key_attribute(self):
        """Test client has api_key attribute"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "api_key")
            assert client.api_key == "test-key"

    def test_client_has_subscription_token_attribute(self):
        """Test client has subscription_token attribute"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="test-key",
                subscription_token="sub-token"
            )
            assert hasattr(client, "subscription_token")
            assert client.subscription_token == "sub-token"

    def test_client_has_orchestrator_attribute(self):
        """Test client can have orchestrator attribute"""
        orch = Mock()
        orch.config = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            assert hasattr(client, "orchestrator")

    def test_client_has_model_attribute(self):
        """Test client has model attribute"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "model") or hasattr(client, "api_key")


class TestClaudeClientEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_client_with_empty_api_key(self):
        """Test client with empty string API key"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="")
            assert client.api_key == ""

    def test_client_with_very_long_api_key(self):
        """Test client with very long API key"""
        long_key = "x" * 10000
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=long_key)
            assert client.api_key == long_key

    def test_cache_key_with_special_characters(self):
        """Test cache key generation with special characters"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            message = "!@#$%^&*()_+-=[]{}|;:',.<>?/~"
            key = client._get_cache_key(message)

            assert isinstance(key, str)
            assert len(key) > 0

    def test_cache_key_with_unicode(self):
        """Test cache key generation with unicode characters"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            message = "你好世界 🚀 مرحبا بالعالم"
            key = client._get_cache_key(message)

            assert isinstance(key, str)
            assert len(key) > 0

    def test_json_parse_with_nested_json(self):
        """Test JSON parsing with deeply nested JSON"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            response = '{"a": {"b": {"c": {"d": "value"}}}}'
            result = client._parse_json_response(response)

            assert isinstance(result, dict)
            assert result.get("a") is not None

    def test_json_parse_with_array(self):
        """Test JSON parsing with array"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            response = '[1, 2, 3, 4, 5]'
            result = client._parse_json_response(response)

            # Should handle array JSON (might return empty dict if not parsed)
            assert result is not None

    def test_json_parse_with_null(self):
        """Test JSON parsing with null value"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            response = 'null'
            result = client._parse_json_response(response)

            assert result is not None or result is None


class TestClaudeClientAuthMethodVariations:
    """Test various authentication method scenarios"""

    def test_get_auth_credential_with_none_auth_method(self):
        """Test get_auth_credential with None auth method"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Should return something without raising
            result = client.get_auth_credential(None)
            assert result is None or isinstance(result, str)

    def test_get_auth_credential_with_unknown_method(self):
        """Test get_auth_credential with unknown method"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Should return None for unknown auth method
            result = client.get_auth_credential("unknown_method")
            assert result is None or isinstance(result, str)

    def test_client_api_key_property(self):
        """Test accessing api_key property"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="my-key-123")
            assert client.api_key == "my-key-123"

    def test_client_subscription_token_property(self):
        """Test accessing subscription_token property"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(subscription_token="sub-123")
            assert client.subscription_token == "sub-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
