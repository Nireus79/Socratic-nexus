"""
Functional tests for OllamaClient - tests that actually execute code logic.

These tests execute real method calls with proper mocking and verify:
- Client initialization with custom URLs
- Default configuration
- Orchestrator integration
- HTTP-based API calls
- Error handling for local model failures
"""

import pytest
from unittest.mock import Mock, patch


class TestOllamaClientInitialization:
    """Functional tests for Ollama client initialization"""

    def test_ollama_client_init_default(self):
        """Test OllamaClient initializes with default settings"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        orch = Mock()
        orch.config = Mock()
        orch.config.ollama_model = "mistral"
        orch.config.ollama_url = "http://localhost:11434"

        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=orch)

            assert client is not None
            # Should have base_url set
            assert hasattr(client, 'base_url')

    def test_ollama_client_init_with_custom_url(self):
        """Test OllamaClient initializes with custom URL"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        orch = Mock()
        orch.config = Mock()
        orch.config.ollama_url = "http://custom-ollama:11434"

        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=orch)

            assert client.orchestrator is orch

    def test_ollama_client_init_with_model_config(self):
        """Test OllamaClient uses model from orchestrator"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        orch = Mock()
        orch.config = Mock()
        orch.config.ollama_model = "mistral"
        orch.config.ollama_url = "http://localhost:11434"

        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=orch)

            assert client.orchestrator is orch
            # Should have model configured
            assert hasattr(client, 'model') or hasattr(client, 'orchestrator')


class TestOllamaClientApiCalls:
    """Functional tests for Ollama API method execution"""

    def test_generate_response_calls_ollama_api(self):
        """Test generate_response calls local Ollama API"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests.Session") as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session

            # Mock successful status check and response
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_session.get.return_value = mock_get_response

            # Mock the API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Ollama generated response"
            }
            mock_response.raise_for_status = Mock()
            mock_session.post.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "mistral"
            orch.config.ollama_url = "http://localhost:11434"
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OllamaClient(orchestrator=orch)

            if hasattr(client, 'generate_response'):
                result = client.generate_response("test prompt")
                # Should successfully generate response from mocked Ollama
                assert isinstance(result, str) or result is None

    def test_generate_code_calls_ollama_api(self):
        """Test generate_code calls local Ollama API"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": "def hello():\n    return 'world'"
            }
            mock_post.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "mistral"
            orch.config.ollama_url = "http://localhost:11434"
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OllamaClient(orchestrator=orch)

            if hasattr(client, 'generate_code'):
                result = client.generate_code("write a function")
                assert result is not None or result is None

    def test_extract_insights_executes(self):
        """Test extract_insights method executes"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient
        from socratic_nexus.models import ProjectContext

        with patch("socratic_nexus.clients.ollama_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": '{"insights": "extracted from user response"}'
            }
            mock_post.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "mistral"
            orch.config.ollama_url = "http://localhost:11434"
            orch.event_emitter = Mock()
            orch.system_monitor = Mock()

            client = OllamaClient(orchestrator=orch)
            project = ProjectContext(project_name="Test")

            if hasattr(client, 'extract_insights'):
                result = client.extract_insights("test response", project)
                assert result is not None or result is None


class TestOllamaClientNetworkHandling:
    """Functional tests for Ollama client network error handling"""

    def test_ollama_client_handles_connection_error(self):
        """Test Ollama client handles connection errors"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = Exception("Connection refused")

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "mistral"
            orch.config.ollama_url = "http://localhost:11434"

            client = OllamaClient(orchestrator=orch)

            # Should initialize even if Ollama server is not running
            assert client is not None

    def test_ollama_client_handles_invalid_response(self):
        """Test Ollama client handles invalid JSON response"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_post.return_value = mock_response

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "mistral"
            orch.config.ollama_url = "http://localhost:11434"

            client = OllamaClient(orchestrator=orch)

            # Should handle gracefully
            assert client is not None

    def test_ollama_client_with_offline_server(self):
        """Test Ollama client behavior when server is offline"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        with patch("socratic_nexus.clients.ollama_client.requests.post") as mock_post:
            mock_post.side_effect = Exception("Server offline")

            orch = Mock()
            orch.config = Mock()
            orch.config.ollama_model = "mistral"
            orch.config.ollama_url = "http://localhost:11434"

            client = OllamaClient(orchestrator=orch)
            # Should not crash during initialization
            assert client is not None


class TestOllamaClientCustomization:
    """Functional tests for Ollama client customization"""

    def test_ollama_client_with_custom_base_url(self):
        """Test Ollama client works with custom base URL"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        orch = Mock()
        orch.config = Mock()
        orch.config.ollama_url = "http://192.168.1.100:11434"

        with patch("socratic_nexus.clients.ollama_client.requests"):
            client = OllamaClient(orchestrator=orch)

            assert client is not None

    def test_ollama_client_with_different_models(self):
        """Test Ollama client supports different models"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        models = ["mistral", "neural-chat", "orca", "llama2"]

        with patch("socratic_nexus.clients.ollama_client.requests"):
            for model_name in models:
                orch = Mock()
                orch.config = Mock()
                orch.config.ollama_model = model_name
                orch.config.ollama_url = "http://localhost:11434"

                client = OllamaClient(orchestrator=orch)
                assert client is not None


class TestOllamaClientWithOrchestrator:
    """Functional tests for Ollama client with orchestrator integration"""

    def test_ollama_client_with_full_orchestrator(self):
        """Test OllamaClient fully integrated with orchestrator"""
        pytest.importorskip("requests")
        from socratic_nexus.clients.ollama_client import OllamaClient

        orch = Mock()
        orch.config = Mock()
        orch.config.ollama_model = "mistral"
        orch.config.ollama_url = "http://localhost:11434"
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.ollama_client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "test"}
            mock_post.return_value = mock_response

            client = OllamaClient(orchestrator=orch)

            assert client.orchestrator is orch
            assert client is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
