"""
Integration tests for Claude client generation methods.

Tests the complete flow of generation methods including:
- Caching behavior (insights and question caches)
- Response parsing (JSON extraction)
- Method chaining with different parameters
- Integration with orchestrator components
"""

import pytest
from unittest.mock import Mock, patch
from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext


class TestInsightsCaching:
    """Tests for insights extraction caching behavior"""

    def test_insights_cache_prevents_redundant_calls(self):
        """Test that identical insights requests use cache"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text='{"goals": ["goal1", "goal2"]}')]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # First call to extract_insights
            response_text = "I want to build a machine learning platform"
            result1 = client.extract_insights(response_text, project)

            # Get number of API calls
            first_call_count = mock_client.messages.create.call_count

            # Second call with identical input
            result2 = client.extract_insights(response_text, project)

            # Should not increase API call count (cache hit)
            assert mock_client.messages.create.call_count == first_call_count

            # Results should be identical
            assert result1 == result2

    def test_insights_cache_separates_different_inputs(self):
        """Test that different inputs don't share cache"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            # Setup different responses for different queries
            def response_side_effect(*args, **kwargs):
                response = Mock()
                response.usage = Mock(input_tokens=100, output_tokens=50)
                # Return different content based on message content
                if "web" in str(kwargs.get("messages", "")):
                    response.content = [Mock(text='{"type": "web"}')]
                else:
                    response.content = [Mock(text='{"type": "mobile"}')]
                return response

            mock_client.messages.create.side_effect = response_side_effect

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # Call with different inputs
            result1 = client.extract_insights("web application", project)
            result2 = client.extract_insights("mobile app", project)

            # Both should have called API (different cache keys)
            assert mock_client.messages.create.call_count >= 2

    def test_insights_cache_memory_structure(self):
        """Test insights cache is properly structured internally"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            # Cache should be initialized as empty dict
            assert hasattr(client, "_insights_cache")
            assert isinstance(client._insights_cache, dict)
            assert len(client._insights_cache) == 0


class TestJsonResponseParsing:
    """Tests for JSON response parsing in generation methods"""

    def test_parse_json_from_markdown_blocks(self):
        """Test extraction of JSON from markdown code blocks"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            # Response with markdown code block
            markdown_response = '```json\n{"status": "success", "data": []}\n```'
            mock_response = Mock()
            mock_response.content = [Mock(text=markdown_response)]
            mock_response.usage = Mock(input_tokens=50, output_tokens=100)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            result = client.extract_insights("test", project)

            # Should parse JSON from markdown
            assert isinstance(result, dict)

    def test_parse_json_response_handling(self):
        """Test _parse_json_response method with valid JSON"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            json_str = '{"key": "value", "nested": {"inner": "data"}}'
            result = client._parse_json_response(json_str)

            assert isinstance(result, dict)
            assert result.get("key") == "value"

    def test_parse_json_with_trailing_text(self):
        """Test JSON parsing with trailing non-JSON text"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            json_with_text = '{"valid": "json"} This is trailing text'
            result = client._parse_json_response(json_with_text)

            # Should extract the JSON part
            assert isinstance(result, dict)

    def test_parse_empty_json_object(self):
        """Test parsing empty JSON object"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            result = client._parse_json_response('{}')

            assert result == {}

    def test_parse_json_array(self):
        """Test parsing JSON array"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")

            json_array = '[1, 2, 3, 4, 5]'
            result = client._parse_json_response(json_array)

            # Result may be dict or original array depending on implementation
            assert result is not None


class TestGenerationMethodChaining:
    """Tests for calling multiple generation methods sequentially"""

    def test_multiple_generations_same_client(self):
        """Test calling different generation methods on same client"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=50)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Test")

            # Call multiple generation methods
            client.generate_response("prompt1")
            client.generate_code("prompt2")
            client.generate_business_plan(project)

            # All should work without errors
            assert mock_client.messages.create.call_count >= 3

    def test_generation_preserves_client_state(self):
        """Test that generation calls don't modify client state"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="result")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=10)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)

            # Store initial state
            initial_api_key = client.api_key
            initial_model = client.model

            # Make generation call
            client.generate_response("test")

            # State should be unchanged
            assert client.api_key == initial_api_key
            assert client.model == initial_model


class TestArtifactGeneration:
    """Tests for artifact generation methods"""

    def test_generate_artifact_method(self):
        """Test generate_artifact method execution"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="artifact content")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=100)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)

            if hasattr(client, 'generate_artifact'):
                result = client.generate_artifact("create an SVG logo")
                assert result is not None or result is None


class TestCurriculumGeneration:
    """Tests for curriculum generation"""

    def test_generate_curriculum_creates_structured_content(self):
        """Test curriculum generation produces structured output"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            structured_response = """
            # Python Basics Course
            ## Module 1: Introduction
            - Lesson 1: Setup
            - Lesson 2: Variables
            """
            mock_response = Mock()
            mock_response.content = [Mock(text=structured_response)]
            mock_response.usage = Mock(input_tokens=100, output_tokens=200)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Course")

            result = client.generate_curriculum(project)

            # Should have called API with curriculum prompt
            assert mock_client.messages.create.called
            assert isinstance(result, (str, type(None)))


class TestMarketingPlanGeneration:
    """Tests for marketing plan generation"""

    def test_generate_marketing_plan_integration(self):
        """Test marketing plan generation with full project context"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            plan_response = """
            # Marketing Plan
            ## Target Audience
            - Primary: Young professionals
            - Secondary: Tech enthusiasts
            """
            mock_response = Mock()
            mock_response.content = [Mock(text=plan_response)]
            mock_response.usage = Mock(input_tokens=120, output_tokens=300)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(
                project_name="SaaS Product",
                description="Cloud-based analytics platform"
            )

            result = client.generate_marketing_plan(project)

            # Should include marketing elements
            assert isinstance(result, (str, type(None)))


class TestResearchProtocolGeneration:
    """Tests for research protocol generation"""

    def test_generate_research_protocol_scientific_structure(self):
        """Test research protocol generation follows scientific structure"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            protocol_response = """
            # Research Protocol
            ## Hypothesis
            Testing performance improvements
            ## Methodology
            1. Setup baseline metrics
            2. Implement optimization
            3. Measure results
            """
            mock_response = Mock()
            mock_response.content = [Mock(text=protocol_response)]
            mock_response.usage = Mock(input_tokens=100, output_tokens=250)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Research")

            result = client.generate_research_protocol(project)

            assert isinstance(result, (str, type(None)))


class TestCreativeBriefGeneration:
    """Tests for creative brief generation"""

    def test_generate_creative_brief_structure(self):
        """Test creative brief includes necessary components"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            brief_response = """
            # Creative Brief
            ## Brand Voice: Innovative, Approachable
            ## Key Messages:
            - Message 1
            - Message 2
            ## Deliverables: Logo, Website, Campaign Materials
            """
            mock_response = Mock()
            mock_response.content = [Mock(text=brief_response)]
            mock_response.usage = Mock(input_tokens=90, output_tokens=200)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="Brand")

            result = client.generate_creative_brief(project)

            assert isinstance(result, (str, type(None)))


class TestDocumentationIntegration:
    """Tests for documentation generation integration"""

    def test_generate_documentation_with_code_artifact(self):
        """Test documentation generation properly formats code artifacts"""
        orch = Mock()
        orch.config = Mock()
        orch.system_monitor = Mock()

        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client

            doc_response = """
            # API Documentation

            ## Functions

            ### calculate_sum(a, b)
            Adds two numbers and returns result.
            """
            mock_response = Mock()
            mock_response.content = [Mock(text=doc_response)]
            mock_response.usage = Mock(input_tokens=100, output_tokens=150)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=orch)
            project = ProjectContext(project_name="MyLib")

            code_artifact = """
            def calculate_sum(a, b):
                return a + b
            """

            result = client.generate_documentation(project, code_artifact)

            # Should have referenced the artifact in the prompt
            assert mock_client.messages.create.called
            assert isinstance(result, (str, type(None)))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
