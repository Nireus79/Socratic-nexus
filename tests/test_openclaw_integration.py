"""Tests for Openclaw integration."""

import pytest
from unittest.mock import Mock

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.clients.openai_client import OpenAIClient
from socratic_nexus.integrations.openclaw import (
    NexusLLMSkill,
    NexusAnalysisSkill,
    NexusCodeGenSkill,
    NexusDocumentationSkill,
)
from socratic_nexus.models import ChatResponse, TokenUsage


@pytest.fixture
def mock_client():
    """Create a mock Socratic Nexus client."""
    client = Mock(spec=ClaudeClient)
    client.model = "claude-3-5-sonnet-20241022"
    client.generate_response = Mock(return_value="Skill response")
    return client


@pytest.fixture
def mock_chat_response_client():
    """Create a mock client that returns ChatResponse."""
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
        content="Skill response from ChatResponse",
        usage=usage,
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
    )
    client.generate_response = Mock(return_value=response)
    return client


class TestNexusLLMSkillInitialization:
    """Test NexusLLMSkill initialization."""

    def test_basic_initialization(self, mock_client):
        """Test basic skill initialization."""
        skill = NexusLLMSkill(client=mock_client)
        assert skill.client == mock_client
        assert skill.name == "nexus_llm_skill"
        assert skill.model == "claude-3-5-sonnet-20241022"

    def test_initialization_with_custom_name(self, mock_client):
        """Test initialization with custom name."""
        skill = NexusLLMSkill(
            client=mock_client,
            name="my_skill",
            description="My custom skill",
        )
        assert skill.name == "my_skill"
        assert skill.description == "My custom skill"

    def test_initialization_with_system_prompt(self, mock_client):
        """Test initialization with system prompt."""
        prompt = "You are an expert analyst."
        skill = NexusLLMSkill(
            client=mock_client,
            system_prompt=prompt,
        )
        assert skill.system_prompt == prompt

    def test_initialization_with_parameters(self, mock_client):
        """Test initialization with all parameters."""
        skill = NexusLLMSkill(
            client=mock_client,
            name="test_skill",
            description="Test description",
            system_prompt="System prompt",
            temperature=0.8,
            max_tokens=512,
            metadata={"version": "1.0"},
        )
        assert skill.temperature == 0.8
        assert skill.max_tokens == 512
        assert skill.metadata["version"] == "1.0"

    def test_initialization_without_client(self):
        """Test initialization without client raises error."""
        with pytest.raises(ValueError, match="client parameter is required"):
            NexusLLMSkill(client=None)

    def test_initialization_with_invalid_client(self):
        """Test initialization with invalid client type."""
        with pytest.raises(ValueError, match="must be a Socratic Nexus client"):
            NexusLLMSkill(client="not a client")


class TestNexusLLMSkillQuery:
    """Test query interface."""

    def test_basic_query(self, mock_client):
        """Test basic query method."""
        skill = NexusLLMSkill(client=mock_client)
        response = skill.query("What is machine learning?")
        assert response == "Skill response"
        mock_client.generate_response.assert_called_once()

    def test_query_with_system_prompt(self, mock_client):
        """Test query includes system prompt."""
        skill = NexusLLMSkill(
            client=mock_client,
            system_prompt="You are an expert.",
        )
        skill.query("Question?")

        call_args = mock_client.generate_response.call_args[0][0]
        assert "You are an expert." in call_args
        assert "Question?" in call_args

    def test_query_returns_chat_response_content(self, mock_chat_response_client):
        """Test query extracts content from ChatResponse."""
        skill = NexusLLMSkill(client=mock_chat_response_client)
        response = skill.query("Test")
        assert response == "Skill response from ChatResponse"

    def test_query_call_count(self, mock_client):
        """Test query increments call count."""
        skill = NexusLLMSkill(client=mock_client)
        assert skill._call_count == 0

        skill.query("Q1")
        assert skill._call_count == 1

        skill.query("Q2")
        assert skill._call_count == 2

    def test_query_error_handling(self, mock_client):
        """Test query handles errors."""
        mock_client.generate_response.side_effect = Exception("API Error")
        skill = NexusLLMSkill(client=mock_client)

        with pytest.raises(Exception, match="API Error"):
            skill.query("Test")


class TestNexusLLMSkillProcess:
    """Test structured process interface."""

    def test_process_with_prompt(self, mock_client):
        """Test process method with prompt."""
        skill = NexusLLMSkill(client=mock_client)
        result = skill.process({"prompt": "Test prompt"})

        assert result["success"] is True
        assert result["output"] == "Skill response"
        assert "metadata" in result

    def test_process_with_input_field(self, mock_client):
        """Test process method with input field."""
        skill = NexusLLMSkill(client=mock_client)
        result = skill.process({"input": "Test input"})

        assert result["success"] is True
        assert result["output"] == "Skill response"

    def test_process_with_context(self, mock_client):
        """Test process with context."""
        skill = NexusLLMSkill(client=mock_client)
        result = skill.process({
            "prompt": "Question?",
            "context": "Background info",
        })

        assert result["success"] is True
        call_args = mock_client.generate_response.call_args[0][0]
        assert "Background info" in call_args

    def test_process_with_instructions(self, mock_client):
        """Test process with special instructions."""
        skill = NexusLLMSkill(client=mock_client)
        result = skill.process({
            "prompt": "Q",
            "instructions": "Answer briefly",
        })

        assert result["success"] is True
        call_args = mock_client.generate_response.call_args[0][0]
        assert "Answer briefly" in call_args

    def test_process_without_prompt(self, mock_client):
        """Test process without prompt returns error."""
        skill = NexusLLMSkill(client=mock_client)
        result = skill.process({})

        assert result["success"] is False
        assert "error" in result

    def test_process_error_handling(self, mock_client):
        """Test process handles errors."""
        mock_client.generate_response.side_effect = Exception("Error")
        skill = NexusLLMSkill(client=mock_client)

        result = skill.process({"prompt": "Test"})
        assert result["success"] is False
        assert "error" in result


class TestNexusLLMSkillInfo:
    """Test skill information methods."""

    def test_get_info(self, mock_client):
        """Test getting skill information."""
        skill = NexusLLMSkill(
            client=mock_client,
            name="test_skill",
            description="Test description",
        )
        info = skill.get_info()

        assert info["name"] == "test_skill"
        assert info["description"] == "Test description"
        assert info["type"] == "llm"
        assert "capabilities" in info

    def test_get_info_capabilities(self, mock_client):
        """Test skill capabilities."""
        skill = NexusLLMSkill(client=mock_client)
        info = skill.get_info()

        expected_capabilities = [
            "text_generation",
            "analysis",
            "code_generation",
            "question_answering",
        ]
        assert all(cap in info["capabilities"] for cap in expected_capabilities)

    def test_string_representation(self, mock_client):
        """Test string representation."""
        skill = NexusLLMSkill(client=mock_client, name="my_skill")
        str_repr = str(skill)

        assert "NexusLLMSkill" in str_repr
        assert "my_skill" in str_repr


class TestSpecializedSkills:
    """Test specialized skill subclasses."""

    def test_analysis_skill_initialization(self, mock_client):
        """Test NexusAnalysisSkill initialization."""
        skill = NexusAnalysisSkill(client=mock_client)
        assert skill.name == "nexus_analysis_skill"
        assert "analyze" in skill.description.lower() or "analysis" in skill.description.lower()

    def test_analysis_skill_analyze_method(self, mock_client):
        """Test analyze method."""
        skill = NexusAnalysisSkill(client=mock_client)
        result = skill.analyze("Sample text")

        assert "original_text" in result
        assert "analysis" in result
        assert result["original_text"] == "Sample text"
        assert result["analysis"] == "Skill response"

    def test_codegen_skill_initialization(self, mock_client):
        """Test NexusCodeGenSkill initialization."""
        skill = NexusCodeGenSkill(client=mock_client)
        assert skill.name == "nexus_codegen_skill"
        assert "code" in skill.description.lower()

    def test_codegen_skill_generate_code(self, mock_client):
        """Test code generation."""
        skill = NexusCodeGenSkill(client=mock_client)
        result = skill.generate_code("Function to add numbers", language="python")

        assert "specification" in result
        assert "language" in result
        assert "code" in result
        assert result["language"] == "python"
        assert result["code"] == "Skill response"

    def test_documentation_skill_initialization(self, mock_client):
        """Test NexusDocumentationSkill initialization."""
        skill = NexusDocumentationSkill(client=mock_client)
        assert skill.name == "nexus_docs_skill"
        assert "documentation" in skill.description.lower()

    def test_documentation_skill_generate_docs(self, mock_client):
        """Test documentation generation."""
        skill = NexusDocumentationSkill(client=mock_client)
        result = skill.generate_documentation(
            "REST API design",
            doc_type="guide"
        )

        assert "subject" in result
        assert "type" in result
        assert "documentation" in result
        assert result["subject"] == "REST API design"
        assert result["type"] == "guide"


class TestNexusSkillMultiProvider:
    """Test skills with different client types."""

    def test_skill_with_openai_client(self):
        """Test skill works with OpenAI client."""
        mock_client = Mock(spec=OpenAIClient)
        mock_client.model = "gpt-4"
        mock_client.generate_response = Mock(return_value="GPT response")

        skill = NexusLLMSkill(client=mock_client)
        response = skill.query("Test")

        assert response == "GPT response"
        assert skill.model == "gpt-4"

    def test_skill_with_different_models(self, mock_client):
        """Test skills with different model configurations."""
        mock_client.model = "claude-3-5-haiku-20241022"
        skill = NexusLLMSkill(client=mock_client)

        assert skill.model == "claude-3-5-haiku-20241022"


class TestNexusSkillAsync:
    """Test async operations."""

    pass  # Async tests require proper async mock setup


class TestSkillIntegration:
    """Integration tests for skill usage."""

    def test_sequential_skill_queries(self, mock_client):
        """Test sequential skill queries."""
        skill = NexusLLMSkill(client=mock_client)

        results = [
            skill.query("Q1"),
            skill.query("Q2"),
            skill.query("Q3"),
        ]

        assert all(r == "Skill response" for r in results)
        assert skill._call_count == 3

    def test_skill_workflow(self, mock_client):
        """Test skill in a workflow."""
        mock_client.generate_response.side_effect = [
            "Analyzed content",
            "Summary of content",
        ]

        analysis_skill = NexusAnalysisSkill(client=mock_client)
        docs_skill = NexusDocumentationSkill(client=mock_client)

        # Analysis step
        analysis = analysis_skill.analyze("Document text")
        assert analysis["analysis"] == "Analyzed content"

        # Documentation step
        docs = docs_skill.generate_documentation("Topic")
        assert docs["documentation"] == "Summary of content"

    def test_multiple_skill_instances(self, mock_client):
        """Test multiple skill instances are independent."""
        skill1 = NexusLLMSkill(client=mock_client, name="skill1")
        skill2 = NexusLLMSkill(client=mock_client, name="skill2")

        assert skill1.name != skill2.name
        assert skill1._call_count == 0
        assert skill2._call_count == 0

        skill1.query("Test")
        assert skill1._call_count == 1
        assert skill2._call_count == 0
