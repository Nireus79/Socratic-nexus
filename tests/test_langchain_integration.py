"""Tests for LangChain integration."""

import pytest
from unittest.mock import Mock

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.clients.openai_client import OpenAIClient
from socratic_nexus.integrations.langchain import SocratesNexusLLM
from socratic_nexus.models import ChatResponse, TokenUsage


@pytest.fixture
def mock_claude_client():
    """Create a mock Claude client."""
    client = Mock(spec=ClaudeClient)
    client.model = "claude-3-5-sonnet-20241022"
    client.generate_response = Mock(return_value="This is a test response from Claude.")
    return client


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = Mock(spec=OpenAIClient)
    client.model = "gpt-4"
    client.generate_response = Mock(return_value="This is a test response from GPT-4.")
    return client


@pytest.fixture
def mock_chat_response_client():
    """Create a mock client that returns ChatResponse object."""
    client = Mock(spec=ClaudeClient)
    client.model = "claude-3-5-sonnet-20241022"
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        cost_usd=0.015,
    )
    response = ChatResponse(
        content="This is a mock ChatResponse.",
        usage=usage,
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
    )
    client.generate_response = Mock(return_value=response)
    return client


class TestSocratesNexusLLMInitialization:
    """Test SocratesNexusLLM initialization."""

    def test_initialization_with_client(self, mock_claude_client):
        """Test initialization with a valid client."""
        llm = SocratesNexusLLM(client=mock_claude_client)
        assert llm.client == mock_claude_client
        assert llm.model == "claude-3-5-sonnet-20241022"
        assert llm.temperature == 0.7

    def test_initialization_with_custom_temperature(self, mock_claude_client):
        """Test initialization with custom temperature."""
        llm = SocratesNexusLLM(client=mock_claude_client, temperature=0.9)
        assert llm.temperature == 0.9

    def test_initialization_with_max_tokens(self, mock_claude_client):
        """Test initialization with max_tokens."""
        llm = SocratesNexusLLM(client=mock_claude_client, max_tokens=1024)
        assert llm.max_tokens == 1024

    def test_initialization_without_client(self):
        """Test initialization without client raises error."""
        with pytest.raises(ValueError, match="client parameter is required"):
            SocratesNexusLLM(client=None)

    def test_initialization_with_invalid_client(self):
        """Test initialization with invalid client type raises error."""
        with pytest.raises(ValueError, match="must be a Socratic Nexus client"):
            SocratesNexusLLM(client="not a client")


class TestSocratesNexusLLMGeneration:
    """Test text generation with SocratesNexusLLM."""

    def test_generate_returns_string(self, mock_claude_client):
        """Test _generate returns a string response."""
        llm = SocratesNexusLLM(client=mock_claude_client)
        response = llm._generate("What is AI?")
        assert isinstance(response, str)
        assert response == "This is a test response from Claude."
        mock_claude_client.generate_response.assert_called_once()

    def test_generate_with_chat_response_object(self, mock_chat_response_client):
        """Test _generate handles ChatResponse objects."""
        llm = SocratesNexusLLM(client=mock_chat_response_client)
        response = llm._generate("What is AI?")
        assert isinstance(response, str)
        assert response == "This is a mock ChatResponse."

    def test_call_invokes_generate(self, mock_claude_client):
        """Test __call__ invokes _generate."""
        llm = SocratesNexusLLM(client=mock_claude_client)
        response = llm("What is AI?")
        assert response == "This is a test response from Claude."
        mock_claude_client.generate_response.assert_called_once()

    def test_generate_with_stop_parameter(self, mock_claude_client):
        """Test _generate with stop parameter."""
        llm = SocratesNexusLLM(client=mock_claude_client)
        response = llm._generate("What is AI?", stop=["END"])
        assert response == "This is a test response from Claude."

    def test_call_count_increments(self, mock_claude_client):
        """Test that call count increments."""
        llm = SocratesNexusLLM(client=mock_claude_client)
        assert llm._call_count == 0
        llm._generate("Test 1")
        assert llm._call_count == 1
        llm._generate("Test 2")
        assert llm._call_count == 2


class TestSocratesNexusLLMAsyncGeneration:
    """Test async text generation."""

    pass  # Async tests require proper async mock setup


class TestSocratesNexusLLMProperties:
    """Test properties and metadata."""

    def test_llm_type_property(self, mock_claude_client):
        """Test _llm_type property."""
        llm = SocratesNexusLLM(client=mock_claude_client)
        assert "socratic_nexus" in llm._llm_type
        assert "mock" in llm._llm_type.lower()

    def test_identifying_params(self, mock_claude_client):
        """Test _identifying_params property."""
        llm = SocratesNexusLLM(client=mock_claude_client, temperature=0.8, max_tokens=512)
        params = llm._identifying_params
        assert params["model"] == "claude-3-5-sonnet-20241022"
        assert params["temperature"] == 0.8
        assert params["max_tokens"] == 512

    def test_get_num_tokens(self, mock_claude_client):
        """Test token counting heuristic."""
        llm = SocratesNexusLLM(client=mock_claude_client)
        # Simple heuristic: ~4 chars per token
        num_tokens = llm.get_num_tokens("This is a test")
        assert num_tokens > 0
        assert isinstance(num_tokens, int)

    def test_get_token_ids(self, mock_claude_client):
        """Test token ID generation."""
        llm = SocratesNexusLLM(client=mock_claude_client)
        token_ids = llm.get_token_ids("This is a test")
        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)


class TestSocratesNexusLLMMultiProvider:
    """Test with multiple provider types."""

    def test_with_openai_client(self, mock_openai_client):
        """Test wrapper works with OpenAI client."""
        llm = SocratesNexusLLM(client=mock_openai_client)
        response = llm._generate("Test prompt")
        assert response == "This is a test response from GPT-4."
        # Should contain "socratic_nexus" and reference to Mock
        assert "socratic_nexus" in llm._llm_type.lower()

    def test_with_different_models(self, mock_claude_client):
        """Test with different model configurations."""
        # Test with default model
        llm1 = SocratesNexusLLM(client=mock_claude_client)
        assert llm1.model == "claude-3-5-sonnet-20241022"

        # Test with model from client
        mock_claude_client.model = "claude-3-5-haiku-20241022"
        llm2 = SocratesNexusLLM(client=mock_claude_client)
        assert llm2.model == "claude-3-5-haiku-20241022"


class TestSocratesNexusLLMErrorHandling:
    """Test error handling."""

    def test_generate_handles_api_error(self, mock_claude_client):
        """Test error handling when API fails."""
        mock_claude_client.generate_response.side_effect = Exception("API Error")
        llm = SocratesNexusLLM(client=mock_claude_client)

        with pytest.raises(Exception, match="API Error"):
            llm._generate("Test prompt")

    def test_invalid_response_type_handling(self, mock_claude_client):
        """Test handling of unexpected response types."""
        # String response should be handled normally
        mock_claude_client.generate_response.return_value = "Direct string"
        llm = SocratesNexusLLM(client=mock_claude_client)
        response = llm._generate("Test")
        assert response == "Direct string"


class TestSocratesNexusLLMIntegration:
    """Integration tests simulating real usage."""

    def test_sequential_calls(self, mock_claude_client):
        """Test multiple sequential calls."""
        llm = SocratesNexusLLM(client=mock_claude_client)

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = [llm._generate(p) for p in prompts]

        assert len(responses) == 3
        assert all(isinstance(r, str) for r in responses)
        assert llm._call_count == 3

    def test_temperature_variation(self, mock_claude_client):
        """Test different temperature settings."""
        temps = [0.1, 0.5, 0.9]
        llms = [SocratesNexusLLM(client=mock_claude_client, temperature=t) for t in temps]

        assert all(isinstance(llm, SocratesNexusLLM) for llm in llms)
        assert [llm.temperature for llm in llms] == temps
