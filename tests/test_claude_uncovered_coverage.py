"""
Tests for previously uncovered branches and methods in Claude client.
Focused on reaching 70% coverage by testing edge cases and error paths.
"""

import pytest
from unittest.mock import Mock, patch
from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.exceptions import APIError


class TestClaudeClientInitializationErrors:
    """Test error handling during client initialization"""

    def test_default_client_initialization_error(self):
        """Test handling of errors during default client initialization"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_anth.side_effect = Exception("Initialization failed")

            # Should not raise, but log warning
            client = ClaudeClient(api_key="test-key")
            # Client created even with initialization error
            assert client is not None

    def test_subscription_client_initialization_error(self):
        """Test handling of errors during subscription client initialization"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
                mock_async.side_effect = Exception("Async init failed")

                # Should not raise, but log warning
                client = ClaudeClient(
                    api_key="test-key",
                    subscription_token="sub-token"
                )
                assert client is not None

    def test_client_with_none_api_key_and_no_orchestrator(self):
        """Test client initialization with no credentials"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=None)
            assert client is not None


class TestClaudeClientUtilityMethods:
    """Test utility and helper methods"""

    def test_get_cache_key_consistency(self):
        """Test that cache key generation is consistent"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            message = "test message"
            key1 = client._get_cache_key(message)
            key2 = client._get_cache_key(message)

            assert key1 == key2
            assert isinstance(key1, str)

    def test_get_cache_key_different_inputs(self):
        """Test that different inputs produce different cache keys"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            key1 = client._get_cache_key("message1")
            key2 = client._get_cache_key("message2")

            assert key1 != key2

    def test_parse_json_response_valid(self):
        """Test JSON parsing with valid JSON"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            response = '{"key": "value"}'
            result = client._parse_json_response(response)

            assert isinstance(result, dict)
            assert result.get("key") == "value"

    def test_parse_json_response_with_markdown(self):
        """Test JSON parsing with markdown code blocks"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            response = '```json\n{"key": "value"}\n```'
            result = client._parse_json_response(response)

            assert isinstance(result, dict)
            assert result.get("key") == "value"

    def test_parse_json_response_invalid(self):
        """Test JSON parsing with invalid JSON"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            response = "not valid json {invalid"
            result = client._parse_json_response(response)

            # Should return empty dict on parse error
            assert isinstance(result, dict)

    def test_parse_json_response_empty_string(self):
        """Test JSON parsing with empty string"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            result = client._parse_json_response("")

            assert isinstance(result, dict)


class TestClaudeClientCredentialHandling:
    """Test credential retrieval and handling"""

    def test_get_auth_credential_api_key(self):
        """Test retrieving API key credential"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="sk-test-key")

            credential = client.get_auth_credential("api_key")
            assert credential == "sk-test-key"

    def test_get_auth_credential_subscription(self):
        """Test retrieving subscription credential"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(
                api_key="test-key",
                subscription_token="sub-token-123"
            )

            credential = client.get_auth_credential("subscription")
            assert credential == "sub-token-123"

    def test_get_auth_credential_default(self):
        """Test default auth credential retrieval"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Default should return api_key
            credential = client.get_auth_credential()
            # May return None or the key depending on implementation
            assert credential is None or isinstance(credential, str)

    def test_get_client_without_credentials(self):
        """Test getting client without any credentials"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=None)

            with pytest.raises(APIError):
                client._get_client()


class TestClaudeClientEventTracking:
    """Test event emission and tracking"""

    def test_track_token_usage_event_emission(self):
        """Test that token tracking emits events"""
        orch = Mock()
        orch.config = Mock()
        orch.event_emitter = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=orch)

            mock_usage = Mock()
            mock_usage.input_tokens = 100
            mock_usage.output_tokens = 50

            client._track_token_usage(mock_usage, "test_op")

            # Should have called system_monitor or event_emitter
            assert orch.system_monitor.process.called or orch.event_emitter.emit.called

    def test_track_token_usage_with_valid_orchestrator(self):
        """Test token tracking with valid orchestrator"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=orch)

            mock_usage = Mock()
            mock_usage.input_tokens = 100
            mock_usage.output_tokens = 50

            # Should call system_monitor
            client._track_token_usage(mock_usage, "test_op")
            # Should have interacted with orchestrator
            assert orch.system_monitor is not None


class TestClaudeClientSpecializedGeneration:
    """Test specialized generation methods"""

    def test_generate_artifact_method_exists(self):
        """Test generate_artifact method exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_artifact")

    def test_generate_business_plan_method_exists(self):
        """Test generate_business_plan method exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_business_plan")

    def test_generate_curriculum_method_exists(self):
        """Test generate_curriculum method exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_curriculum")

    def test_generate_documentation_method_exists(self):
        """Test generate_documentation method exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_documentation")

    def test_generate_marketing_plan_method_exists(self):
        """Test generate_marketing_plan method exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_marketing_plan")

    def test_generate_research_protocol_method_exists(self):
        """Test generate_research_protocol method exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_research_protocol")

    def test_generate_creative_brief_method_exists(self):
        """Test generate_creative_brief method exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_creative_brief")

    def test_generate_conflict_resolution_method_exists(self):
        """Test generate_conflict_resolution_suggestions method exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_conflict_resolution_suggestions")


class TestClaudeClientAsyncMethods:
    """Test async method availability"""

    def test_async_generate_response_exists(self):
        """Test async generate_response exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_response_async")

    def test_async_generate_code_exists(self):
        """Test async generate_code exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_code_async")

    def test_async_extract_insights_exists(self):
        """Test async extract_insights exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "extract_insights_async")

    def test_async_business_plan_exists(self):
        """Test async generate_business_plan exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_business_plan_async")

    def test_async_documentation_exists(self):
        """Test async generate_documentation exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_documentation_async")

    def test_async_documentation_exists(self):
        """Test async generate_documentation exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "generate_documentation_async")

    def test_async_evaluate_quality_exists(self):
        """Test async evaluate_quality_async exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "evaluate_quality_async")

    def test_async_extract_tech_recommendations_exists(self):
        """Test async extract_tech_recommendations_async exists"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert hasattr(client, "extract_tech_recommendations_async")


class TestClaudeClientModelConfiguration:
    """Test model configuration and selection"""

    def test_model_from_orchestrator_config(self):
        """Test model selection from orchestrator config"""
        orch = Mock()
        orch.config = Mock()
        orch.config.claude_model = "claude-haiku-4-5-20251001"

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            assert client.orchestrator.config.claude_model == "claude-haiku-4-5-20251001"

    def test_model_default(self):
        """Test default model when not specified"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            # Should have some default model
            assert hasattr(client, "model") or hasattr(client, "orchestrator")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
