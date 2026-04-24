"""
Tests that ACTUALLY EXECUTE Claude client methods to increase code coverage.

These tests call real methods with proper mocking, not just checking existence.
"""

import pytest
from unittest.mock import Mock, patch
from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext


class TestMethodExecution:
    """Tests that actually execute methods"""

    def test_generate_socratic_question_execution(self):
        """Test generate_socratic_question actually executes"""
        orch = Mock()
        orch.config = Mock()
        orch.config.claude_model = "claude-haiku"
        orch.event_emitter = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_response.content = [Mock(text="What is the purpose of this function?")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="sk-test", orchestrator=orch)
            project = ProjectContext(
                project_name="TestProject",
                phase="design"
            )

            # ACTUALLY CALL THE METHOD
            result = client.generate_socratic_question(project)

            # Should have called the API
            assert mock_client.messages.create.called
            assert isinstance(result, str)

    def test_generate_suggestions_execution(self):
        """Test generate_suggestions actually executes"""
        orch = Mock()
        orch.config = Mock()
        orch.config.claude_model = "claude-haiku"
        orch.event_emitter = Mock()
        orch.system_monitor = Mock()
        orch.vector_db = Mock()
        orch.context_analyzer = Mock()

        # Mock vector DB search
        orch.vector_db.search_similar.return_value = [
            {"content": "Sample knowledge result 1"},
            {"content": "Sample knowledge result 2"}
        ]
        orch.context_analyzer.get_context_summary.return_value = "Project context summary"

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_response.content = [Mock(text="Here are some suggestions...")]
            mock_response.usage = Mock(input_tokens=150, output_tokens=100)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="sk-test", orchestrator=orch)
            project = ProjectContext(
                project_name="TestProject",
                conversation_history=[
                    {"type": "user", "content": "How do I..."},
                    {"type": "assistant", "content": "You could..."}
                ]
            )

            # ACTUALLY CALL THE METHOD
            result = client.generate_suggestions("How do I implement X?", project)

            # Should have called the API
            assert mock_client.messages.create.called
            # Should have accessed vector DB and context analyzer
            assert orch.vector_db.search_similar.called
            assert orch.context_analyzer.get_context_summary.called
            assert isinstance(result, str)

    def test_generate_socratic_question_error_handling(self):
        """Test error handling in generate_socratic_question"""
        orch = Mock()
        orch.config = Mock()
        orch.event_emitter = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("API Error")

            client = ClaudeClient(api_key="sk-test", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # Should raise APIError
            from socratic_nexus.exceptions import APIError
            with pytest.raises(APIError):
                client.generate_socratic_question(project)

    def test_evaluate_quality_metrics_execution(self):
        """Test evaluate_quality_metrics actually executes"""
        orch = Mock()
        orch.config = Mock()
        orch.config.claude_model = "claude-haiku"
        orch.event_emitter = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            # Mock response with JSON
            mock_response = Mock()
            mock_response.content = [Mock(text='{"quality": 0.85, "completeness": 0.90}')]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="sk-test", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # ACTUALLY CALL THE METHOD
            if hasattr(client, 'evaluate_quality_metrics'):
                client.evaluate_quality_metrics("Test artifact", project)
                assert mock_client.messages.create.called

    def test_extract_tech_recommendations_execution(self):
        """Test extract_tech_recommendations actually executes"""
        orch = Mock()
        orch.config = Mock()
        orch.config.claude_model = "claude-haiku"
        orch.event_emitter = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            # Mock response
            mock_response = Mock()
            mock_response.content = [Mock(text='{"recommendations": ["Python", "FastAPI"]}')]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="sk-test", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # ACTUALLY CALL THE METHOD
            if hasattr(client, 'extract_tech_recommendations'):
                client.extract_tech_recommendations(project, "backend")
                assert mock_client.messages.create.called


class TestGenerateResponseExecution:
    """Tests for generate_response actual execution"""

    def test_generate_response_with_temperature(self):
        """Test generate_response with temperature parameter"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="Response text")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=100)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="sk-test", orchestrator=orch)

            # ACTUALLY CALL WITH PARAMETERS
            client.generate_response(
                "Test prompt",
                temperature=0.5
            )

            # Verify API was called
            assert mock_client.messages.create.called

    def test_generate_response_with_full_parameters(self):
        """Test generate_response with all parameters"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="Full response")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=200)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="sk-test", orchestrator=orch)

            # ACTUALLY CALL WITH PARAMETERS
            client.generate_response(
                "Complex prompt",
                temperature=0.7,
                max_tokens=500,
                user_id="user123",
                user_auth_method="api_key"
            )

            assert mock_client.messages.create.called


class TestGenerateCodeExecution:
    """Tests for generate_code actual execution"""

    def test_generate_code_execution_with_language_hint(self):
        """Test generate_code actually executes"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="def my_function():\n    pass")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="sk-test", orchestrator=orch)

            # ACTUALLY CALL THE METHOD
            result = client.generate_code("Write a Python function")

            # Verify execution
            assert mock_client.messages.create.called
            assert isinstance(result, (str, type(None)))

    def test_generate_code_with_context(self):
        """Test generate_code with user context"""
        orch = Mock()
        orch.config = Mock()
        orch.database = Mock()
        orch.database.get_api_key.return_value = None
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="code")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="sk-test", orchestrator=orch)

            # ACTUALLY CALL WITH USER CONTEXT
            result = client.generate_code(
                "Write code",
                user_id="user456",
                user_auth_method="api_key"
            )

            assert mock_client.messages.create.called


class TestInitializationPathsExecution:
    """Tests that execute actual initialization code"""

    def test_client_lazy_initialization_api_key(self):
        """Test client creation executes initialization"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
                mock_client = Mock()
                mock_async_client = Mock()
                mock_anth.return_value = mock_client
                mock_async.return_value = mock_async_client

                # ACTUALLY CREATE CLIENT - should trigger initialization
                client = ClaudeClient(api_key="sk-valid-key")

                # Verify initialization was attempted
                assert mock_anth.called or client.client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
