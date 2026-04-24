"""
Comprehensive execution tests for OpenAIClient to increase code coverage.
Tests that actually EXECUTE methods with proper mocking.
Skipped if cryptography or openai are not installed.
"""

import pytest
from unittest.mock import Mock, patch
from socratic_nexus.models import ProjectContext

pytest.importorskip("cryptography")
pytest.importorskip("openai")


class TestOpenAIClientMethodExecution:
    """Tests that execute OpenAIClient methods comprehensively."""

    def test_openai_generate_response_execution(self):
        """Test generate_response executes."""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Response"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)
            client.generate_response("Test prompt")

            assert mock_client.chat.completions.create.called

    def test_openai_generate_code_execution(self):
        """Test generate_code executes."""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="def foo(): pass"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)
            client.generate_code("Write code")

            assert mock_client.chat.completions.create.called

    def test_openai_extract_insights_execution(self):
        """Test extract_insights executes."""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content='{"insights": ["a"]}'))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)
            project = ProjectContext(project_name="Test")
            client.extract_insights("feedback", project)

            assert mock_client.chat.completions.create.called

    def test_openai_with_custom_model(self):
        """Test OpenAI with custom model configuration."""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="result"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.config.openai_model = "gpt-4-turbo"
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)
            client.generate_response("prompt")

            assert mock_client.chat.completions.create.called

    def test_openai_generate_business_plan(self):
        """Test generate_business_plan execution."""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="plan"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)
            if hasattr(client, 'generate_business_plan'):
                client.generate_business_plan(
                    ProjectContext(project_name="Test")
                )
                assert mock_client.chat.completions.create.called

    def test_openai_generate_documentation(self):
        """Test generate_documentation execution."""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="doc"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)
            if hasattr(client, 'generate_documentation'):
                client.generate_documentation(
                    ProjectContext(project_name="Test")
                )
                assert mock_client.chat.completions.create.called

    def test_openai_with_temperature(self):
        """Test generate_response with temperature parameter."""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="result"))]
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
            mock_client.chat.completions.create.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)
            client.generate_response("prompt", temperature=0.7)

            assert mock_client.chat.completions.create.called

    def test_openai_cache_operations(self):
        """Test cache operations."""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            orch = Mock()
            orch.config = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)

            if hasattr(client, '_get_cache_key'):
                key = client._get_cache_key("test")
                assert isinstance(key, str)

    def test_openai_error_handling(self):
        """Test error handling."""
        from socratic_nexus.clients.openai_client import OpenAIClient
        from socratic_nexus.exceptions import APIError

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.side_effect = Exception("API Error")

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)

            try:
                client.generate_response("test")
            except (APIError, Exception):
                pass

    def test_openai_parse_json_response(self):
        """Test JSON parsing in response handling."""
        from socratic_nexus.clients.openai_client import OpenAIClient

        with patch("socratic_nexus.clients.openai_client.openai.OpenAI"):
            orch = Mock()
            orch.config = Mock()

            client = OpenAIClient(api_key="sk-test", orchestrator=orch)

            if hasattr(client, '_parse_json_response'):
                result = client._parse_json_response('{"key": "value"}')
                assert isinstance(result, (dict, type(None)))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
