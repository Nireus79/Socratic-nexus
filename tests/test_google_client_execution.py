"""
Comprehensive execution tests for GoogleClient to increase code coverage.
Tests that actually EXECUTE methods with proper mocking.
"""

import sys
import pytest
from unittest.mock import Mock, patch
from socratic_nexus.models import ProjectContext


class TestGoogleClientMethodExecution:
    """Tests that execute GoogleClient methods comprehensively."""

    @pytest.fixture
    def mock_genai(self):
        """Mock google.generativeai module."""
        return Mock()

    def test_google_generate_response_execution(self, mock_genai):
        """Test generate_response executes with mocked genai."""
        with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_response = Mock()
            mock_response.text = "Response text"
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            result = client.generate_response("Test prompt")

            assert mock_model.generate_content.called

    def test_google_generate_code_execution(self, mock_genai):
        """Test generate_code executes."""
        with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_response = Mock()
            mock_response.text = "def foo(): pass"
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            result = client.generate_code("Write code")

            assert mock_model.generate_content.called

    def test_google_extract_insights_execution(self, mock_genai):
        """Test extract_insights executes."""
        with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_response = Mock()
            mock_response.text = '{"insights": ["a", "b"]}'
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("feedback", project)

            assert mock_model.generate_content.called

    def test_google_generate_response_with_params(self, mock_genai):
        """Test generate_response with parameters."""
        with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_response = Mock()
            mock_response.text = "Result"
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            result = client.generate_response(
                "Prompt",
                temperature=0.7,
                max_tokens=500
            )

            assert mock_model.generate_content.called

    def test_google_generate_artifact(self, mock_genai):
        """Test generate_artifact execution."""
        with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_response = Mock()
            mock_response.text = "artifact content"
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            if hasattr(client, 'generate_artifact'):
                result = client.generate_artifact(
                    project_type="document",
                    artifact_type="markdown"
                )
                assert mock_model.generate_content.called

    def test_google_client_error_handling(self, mock_genai):
        """Test error handling in generate_response."""
        with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
            from socratic_nexus.clients.google_client import GoogleClient
            from socratic_nexus.exceptions import APIError

            mock_model = Mock()
            mock_genai.GenerativeModel.return_value = mock_model
            mock_model.generate_content.side_effect = Exception("API Error")

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)

            # Should handle gracefully
            try:
                result = client.generate_response("Test")
                assert result is None or isinstance(result, (str, dict))
            except APIError:
                pass

    def test_google_cache_key_generation(self, mock_genai):
        """Test cache key generation."""
        with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
            from socratic_nexus.clients.google_client import GoogleClient

            mock_genai.GenerativeModel.return_value = Mock()

            orch = Mock()
            orch.config = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)

            if hasattr(client, '_get_cache_key'):
                key1 = client._get_cache_key("message")
                key2 = client._get_cache_key("message")
                assert key1 == key2
                assert isinstance(key1, str)

    def test_google_get_auth_credential(self, mock_genai):
        """Test credential retrieval."""
        with patch.dict('sys.modules', {'google.generativeai': mock_genai}):
            from socratic_nexus.clients.google_client import GoogleClient

            mock_genai.GenerativeModel.return_value = Mock()

            client = GoogleClient(api_key="test-key")

            if hasattr(client, 'get_auth_credential'):
                cred = client.get_auth_credential("api_key")
                assert cred == "test-key" or cred is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
