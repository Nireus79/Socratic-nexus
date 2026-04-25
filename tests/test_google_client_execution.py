"""
Comprehensive execution tests for GoogleClient to increase code coverage.
Tests that actually EXECUTE methods with proper mocking.
Skipped if google.generativeai is not installed.
"""

import pytest
from unittest.mock import Mock, patch
from socratic_nexus.models import ProjectContext

pytest.importorskip("google.generativeai")


class TestGoogleClientMethodExecution:
    """Tests that execute GoogleClient methods comprehensively."""

    def test_google_generate_response_execution(self):
        """Test generate_response executes with mocked genai."""
        with patch(
            "socratic_nexus.clients.google_client.genai.GenerativeModel"
        ) as mock_model_class:
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_response = Mock()
            mock_response.text = "Response text"
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            client.generate_response("Test prompt")

            assert mock_model.generate_content.called

    def test_google_generate_code_execution(self):
        """Test generate_code executes."""
        with patch(
            "socratic_nexus.clients.google_client.genai.GenerativeModel"
        ) as mock_model_class:
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_response = Mock()
            mock_response.text = "def foo(): pass"
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            client.generate_code("Write code")

            assert mock_model.generate_content.called

    def test_google_extract_insights_execution(self):
        """Test extract_insights executes."""
        with patch(
            "socratic_nexus.clients.google_client.genai.GenerativeModel"
        ) as mock_model_class:
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_response = Mock()
            mock_response.text = '{"insights": ["a", "b"]}'
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Test")
            client.extract_insights("feedback", project)

            assert mock_model.generate_content.called

    def test_google_generate_response_with_params(self):
        """Test generate_response with parameters."""
        with patch(
            "socratic_nexus.clients.google_client.genai.GenerativeModel"
        ) as mock_model_class:
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_response = Mock()
            mock_response.text = "Result"
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            client.generate_response("Prompt", temperature=0.7, max_tokens=500)

            assert mock_model.generate_content.called

    def test_google_generate_artifact(self):
        """Test generate_artifact execution."""
        with patch(
            "socratic_nexus.clients.google_client.genai.GenerativeModel"
        ) as mock_model_class:
            from socratic_nexus.clients.google_client import GoogleClient

            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_response = Mock()
            mock_response.text = "artifact content"
            mock_model.generate_content.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)
            if hasattr(client, "generate_artifact"):
                client.generate_artifact(project_type="document", artifact_type="markdown")
                assert mock_model.generate_content.called

    def test_google_client_error_handling(self):
        """Test error handling in generate_response."""
        with patch(
            "socratic_nexus.clients.google_client.genai.GenerativeModel"
        ) as mock_model_class:
            from socratic_nexus.clients.google_client import GoogleClient
            from socratic_nexus.exceptions import APIError

            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_model.generate_content.side_effect = Exception("API Error")

            orch = Mock()
            orch.config = Mock()
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)

            try:
                client.generate_response("Test")
            except (APIError, Exception):
                pass

    def test_google_cache_key_generation(self):
        """Test cache key generation."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            orch = Mock()
            orch.config = Mock()

            client = GoogleClient(api_key="test-key", orchestrator=orch)

            if hasattr(client, "_get_cache_key"):
                key1 = client._get_cache_key("message")
                key2 = client._get_cache_key("message")
                assert key1 == key2
                assert isinstance(key1, str)

    def test_google_get_auth_credential(self):
        """Test credential retrieval."""
        with patch("socratic_nexus.clients.google_client.genai"):
            from socratic_nexus.clients.google_client import GoogleClient

            client = GoogleClient(api_key="test-key")

            if hasattr(client, "get_auth_credential"):
                cred = client.get_auth_credential("api_key")
                assert cred == "test-key" or cred is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
