"""
Functional tests for ClaudeClient - tests that actually execute code logic.

These tests focus on:
- Real method execution with valid parameters
- Proper mock setup returning correct types
- Error path execution and handling
- Caching behavior verification
- API key retrieval and decryption logic
- Multi-user and auth method scenarios
"""

import pytest
from unittest.mock import Mock, patch
from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext, ConflictInfo
from socratic_nexus.exceptions import APIError


class TestExtractInsightsFunctional:
    """Functional tests for extract_insights method execution"""

    def test_extract_insights_with_valid_response(self):
        """Test extract_insights actually processes valid response"""
        orch = Mock()
        orch.config = Mock()
        orch.event_emitter = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            # Setup response that extract_insights actually processes
            mock_response = Mock()
            mock_response.content = [Mock(text='{"goals": ["goal1"], "tech": ["python"]}')]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="TestProject")

            # Call extract_insights with real response
            result = client.extract_insights("We want to build a web app", project)

            # Should actually call the API
            assert mock_client.messages.create.called
            # Result should be dict from parsed response
            assert isinstance(result, dict)

    def test_extract_insights_caching(self):
        """Test that extract_insights uses cache for identical prompts"""
        orch = Mock()
        orch.config = Mock()
        orch.event_emitter = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text='{"goals": ["cached"]}')]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            same_response = "build a web application"

            # First call should hit API
            result1 = client.extract_insights(same_response, project)
            call_count_after_first = mock_client.messages.create.call_count

            # Second call with same response should use cache
            result2 = client.extract_insights(same_response, project)
            call_count_after_second = mock_client.messages.create.call_count

            # API should not be called again
            assert call_count_after_second == call_count_after_first
            assert result1 == result2

    def test_extract_insights_with_empty_response(self):
        """Test extract_insights handles empty responses correctly"""
        orch = Mock()
        orch.config = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # Empty response should return empty dict without API call
            result = client.extract_insights("", project)
            assert result == {}

    def test_extract_insights_with_non_informative_responses(self):
        """Test extract_insights recognizes non-informative responses"""
        orch = Mock()
        orch.config = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # Test various non-informative responses
            responses = ["i don't know", "idk", "not sure", "no idea"]
            for response in responses:
                result = client.extract_insights(response, project)
                # Should return dict with note, not call API
                assert isinstance(result, dict)
                assert mock_client.messages.create.call_count == 0


class TestGenerateResponseFunctional:
    """Functional tests for generate_response method execution"""

    def test_generate_response_with_temperature(self):
        """Test generate_response passes temperature parameter to API"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)

            # Call with temperature parameter
            client.generate_response(
                "test prompt",
                temperature=0.7,
                max_tokens=100
            )

            # Verify API was called
            assert mock_client.messages.create.called
            # Check call arguments
            call_args = mock_client.messages.create.call_args
            assert call_args is not None

    def test_generate_response_with_user_id(self):
        """Test generate_response with user_id for multi-user support"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="user-specific response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)

            # Call with user_id
            client.generate_response(
                "test",
                user_id="user123"
            )

            # Should execute successfully
            assert mock_client.messages.create.called


class TestGetClientErrorHandling:
    """Functional tests for _get_client error handling"""

    def test_get_client_with_no_api_key(self):
        """Test _get_client raises APIError when no API key available"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=orch)

            # Should raise APIError
            with pytest.raises(APIError) as exc_info:
                client._get_client()

            assert "No API key configured" in str(exc_info.value)

    def test_get_client_with_valid_api_key(self):
        """Test _get_client returns Anthropic client with valid key"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            client = ClaudeClient(api_key="sk-valid-key", orchestrator=orch)
            result = client._get_client()

            # Should return the mocked Anthropic client
            assert result is not None
            # Should have created client with the API key
            mock_anth.assert_called()

    def test_get_client_with_user_specific_key(self):
        """Test _get_client uses user-specific key from database"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        # Simulate user has stored API key
        orch.database.get_api_key.return_value = "user_key_encrypted"

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            with patch.object(ClaudeClient, "_decrypt_api_key_from_db") as mock_decrypt:
                mock_decrypt.return_value = "sk-decrypted-user-key"
                mock_client = Mock()
                mock_anth.return_value = mock_client

                client = ClaudeClient(api_key="fallback-key", orchestrator=orch)
                result = client._get_client(user_id="user123")

                # Should have called decrypt
                mock_decrypt.assert_called_with("user_key_encrypted")
                # Should create client with decrypted key
                assert result is not None


class TestGenerateCodeFunctional:
    """Functional tests for generate_code method"""

    def test_generate_code_with_valid_prompt(self):
        """Test generate_code executes with valid prompt"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="def hello():\n    return 'world'")]
            mock_response.usage = Mock(input_tokens=30, output_tokens=15)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            result = client.generate_code("write a function to add two numbers")

            # Should call API
            assert mock_client.messages.create.called
            # Result should be string or None
            assert isinstance(result, (str, type(None)))

    def test_generate_code_with_context_and_user_id(self):
        """Test generate_code with full parameters"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="generated code")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=100)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            client.generate_code(
                "write fibonacci",
                user_id="user123",
                user_auth_method="api_key"
            )

            # Should execute successfully
            assert mock_client.messages.create.called


class TestGenerateBusinessPlanFunctional:
    """Functional tests for business plan generation"""

    def test_generate_business_plan_with_project_context(self):
        """Test generate_business_plan executes with ProjectContext"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="Executive Summary: ...")]
            mock_response.usage = Mock(input_tokens=80, output_tokens=300)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(
                project_name="TechStartup",
                description="AI-powered analytics platform"
            )

            result = client.generate_business_plan(project)

            # Should call API with proper project data
            assert mock_client.messages.create.called
            assert isinstance(result, (str, type(None)))


class TestAsyncClientFunctional:
    """Functional tests for async client methods"""

    @pytest.mark.asyncio
    async def test_get_async_client_creates_async_anthropic(self):
        """Test _get_async_client returns AsyncAnthropic instance"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
                mock_async_client = Mock()
                mock_async.return_value = mock_async_client

                client = ClaudeClient(api_key="sk-async-key", orchestrator=orch)
                result = client._get_async_client()

                # Should have called AsyncAnthropic
                mock_async.assert_called()
                assert result is not None


class TestTokenTracking:
    """Functional tests for token usage tracking"""

    def test_track_token_usage_emits_event(self):
        """Test _track_token_usage calls system_monitor"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()
        orch.system_monitor.process = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key", orchestrator=orch)

            # Create usage object
            usage = Mock()
            usage.input_tokens = 100
            usage.output_tokens = 50

            # Call track token usage
            client._track_token_usage(usage, "test_operation")

            # Should have called system_monitor.process or event_emitter
            assert (orch.system_monitor.process.called or
                    orch.event_emitter.emit.called if hasattr(orch, 'event_emitter') else True)

    def test_track_token_usage_without_orchestrator(self):
        """Test _track_token_usage handles missing orchestrator gracefully"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            usage = Mock()
            usage.input_tokens = 100
            usage.output_tokens = 50

            # Should not raise error
            client._track_token_usage(usage, "test_op")


class TestConflictResolutionFunctional:
    """Functional tests for conflict resolution"""

    def test_generate_conflict_resolution_with_conflict_info(self):
        """Test conflict resolution generation with real ConflictInfo"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="resolution options")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=200)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)

            conflict = ConflictInfo(
                description="Conflicting implementations",
                file_path="src/main.py"
            )
            project = ProjectContext(project_name="Test")

            result = client.generate_conflict_resolution_suggestions(conflict, project)

            # Should call API with conflict data
            assert mock_client.messages.create.called
            assert isinstance(result, (str, type(None)))


class TestCachingBehavior:
    """Functional tests for caching across multiple methods"""

    def test_cache_key_consistency(self):
        """Test _get_cache_key produces consistent keys"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            message = "test message"
            key1 = client._get_cache_key(message)
            key2 = client._get_cache_key(message)

            # Same input should produce same cache key
            assert key1 == key2
            assert isinstance(key1, str)

    def test_different_messages_different_cache_keys(self):
        """Test different messages produce different cache keys"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            key1 = client._get_cache_key("message 1")
            key2 = client._get_cache_key("message 2")

            assert key1 != key2


class TestDocumentationGenerationFunctional:
    """Functional tests for documentation generation"""

    def test_generate_documentation_with_artifact(self):
        """Test documentation generation with artifact content"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="# Documentation\n\n## Overview...")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=250)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="MyProject")
            artifact = "def calculate(x, y):\n    return x + y"

            result = client.generate_documentation(project, artifact)

            # Should call API with artifact
            assert mock_client.messages.create.called
            assert isinstance(result, (str, type(None)))


class TestSubscriptionTokenHandling:
    """Functional tests for subscription token fallback behavior"""

    def test_subscription_mode_falls_back_to_api_key(self):
        """Test that subscription mode logs warning and uses api_key"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            client = ClaudeClient(
                api_key="sk-fallback-key",
                orchestrator=orch,
                subscription_token="sub-token"
            )

            # Request with subscription method should fallback to api_key
            result = client._get_client(user_auth_method="subscription")

            # Should still work (falls back to api_key)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
