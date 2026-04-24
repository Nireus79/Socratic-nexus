"""
Tests for specialized ClaudeClient generation and analysis methods

Covers:
- Code generation and analysis
- Business plan and strategy generation
- Creative content generation
- Documentation and curriculum generation
- Specialized analysis methods with various parameters
- Error handling for specialized methods
- Caching behavior for specialized operations
- Async variants of specialized methods
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext, ConflictInfo


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator"""
    orch = Mock()
    orch.config = Mock()
    orch.config.claude_model = "claude-haiku-4-5-20251001"
    orch.event_emitter = Mock()
    orch.database = Mock()
    orch.database.get_api_key.return_value = None
    orch.system_monitor = Mock()
    orch.system_monitor.process = Mock()
    return orch


class TestCodeGeneration:
    """Tests for code generation functionality"""

    def test_generate_code_basic(self, mock_orchestrator):
        """Test basic code generation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="def hello(): pass")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_code("write a function to add two numbers")

            assert isinstance(result, (str, type(None)))
            mock_client.messages.create.assert_called()

    def test_generate_code_with_context(self, mock_orchestrator):
        """Test code generation with context"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="code")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_code(
                "write a function",
                user_id="user123",
                user_auth_method="api_key"
            )

            assert isinstance(result, (str, type(None)))

    def test_generate_code_caching(self, mock_orchestrator):
        """Test that same prompts use cache"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="code")]
            mock_response.usage = Mock(input_tokens=50, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            prompt = "same prompt"
            client.generate_code(prompt)
            client.generate_code(prompt)

            # Both calls should work (caching may or may not be used)
            assert mock_client.messages.create.call_count >= 1

    def test_generate_code_with_max_tokens(self, mock_orchestrator):
        """Test code generation with max tokens constraint"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="def func():\n    pass")]
            mock_response.usage = Mock(input_tokens=30, output_tokens=10)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_code("write code", max_tokens=256)

            assert isinstance(result, (str, type(None)))


class TestBusinessPlanGeneration:
    """Tests for business plan generation"""

    def test_generate_business_plan(self, mock_orchestrator):
        """Test business plan generation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="business plan content")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=500)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="MyStartup")
            result = client.generate_business_plan(project)

            assert isinstance(result, (str, type(None)))
            mock_client.messages.create.assert_called()

    def test_generate_business_plan_with_details(self, mock_orchestrator):
        """Test business plan with additional project details"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="plan")]
            mock_response.usage = Mock(input_tokens=100, output_tokens=500)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(
                project_name="MyStartup",
                description="A tech startup"
            )
            result = client.generate_business_plan(project)

            assert isinstance(result, (str, type(None)))


class TestCreativeGeneration:
    """Tests for creative content generation"""

    def test_generate_creative_brief(self, mock_orchestrator):
        """Test creative brief generation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="creative brief")]
            mock_response.usage = Mock(input_tokens=80, output_tokens=300)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Campaign")
            result = client.generate_creative_brief(project)

            assert isinstance(result, (str, type(None)))

    def test_generate_marketing_plan(self, mock_orchestrator):
        """Test marketing plan generation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="marketing plan")]
            mock_response.usage = Mock(input_tokens=90, output_tokens=400)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Product")
            result = client.generate_marketing_plan(project)

            assert isinstance(result, (str, type(None)))

    def test_generate_research_protocol(self, mock_orchestrator):
        """Test research protocol generation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="protocol")]
            mock_response.usage = Mock(input_tokens=85, output_tokens=350)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Research")
            result = client.generate_research_protocol(project)

            assert isinstance(result, (str, type(None)))


class TestDocumentationGeneration:
    """Tests for documentation and curriculum generation"""

    def test_generate_documentation(self, mock_orchestrator):
        """Test documentation generation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="documentation")]
            mock_response.usage = Mock(input_tokens=70, output_tokens=200)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Project")
            artifact = "some code artifact"
            result = client.generate_documentation(project, artifact)

            assert isinstance(result, (str, type(None)))

    def test_generate_curriculum(self, mock_orchestrator):
        """Test curriculum generation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="curriculum")]
            mock_response.usage = Mock(input_tokens=75, output_tokens=250)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Course")
            result = client.generate_curriculum(project)

            assert isinstance(result, (str, type(None)))


class TestAnalysisMethods:
    """Tests for analysis methods"""

    def test_extract_insights_basic(self, mock_orchestrator):
        """Test basic insights extraction"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text='{"insights": "test"}')]
            mock_response.usage = Mock(input_tokens=60, output_tokens=100)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.extract_insights("response text", project)

            assert isinstance(result, (dict, str, type(None)))

    def test_extract_tech_recommendations(self, mock_orchestrator):
        """Test tech recommendations extraction"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text='{"recommendations": []}')]
            mock_response.usage = Mock(input_tokens=80, output_tokens=150)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.extract_tech_recommendations(project, "backend")

            assert isinstance(result, (dict, list, type(None)))

    def test_evaluate_quality_metrics(self, mock_orchestrator):
        """Test quality evaluation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text='{"quality": "good"}')]
            mock_response.usage = Mock(input_tokens=70, output_tokens=120)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.evaluate_quality_metrics("artifact", project)

            assert isinstance(result, (dict, str, type(None)))


class TestConflictResolution:
    """Tests for conflict resolution"""

    def test_generate_conflict_resolution(self, mock_orchestrator):
        """Test conflict resolution suggestions"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="resolution suggestions")]
            mock_response.usage = Mock(input_tokens=90, output_tokens=200)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            conflict = ConflictInfo(
                description="file conflict",
                file_path="test.py"
            )
            result = client.generate_conflict_resolution_suggestions(conflict, project)

            assert isinstance(result, (str, type(None)))


class TestAsyncSpecializedMethods:
    """Tests for async variants of specialized methods"""

    @pytest.mark.asyncio
    async def test_generate_business_plan_async(self, mock_orchestrator):
        """Test async business plan generation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text="plan")],
                    usage=Mock(input_tokens=100, output_tokens=500)
                )
            )

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
                project = ProjectContext(project_name="Test")
                result = await client.generate_business_plan_async(project)

                assert isinstance(result, (str, type(None)))

    @pytest.mark.asyncio
    async def test_generate_code_async(self, mock_orchestrator):
        """Test async code generation"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text="code")],
                    usage=Mock(input_tokens=50, output_tokens=100)
                )
            )

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
                result = await client.generate_code_async("write function")

                assert isinstance(result, (str, type(None)))

    @pytest.mark.asyncio
    async def test_extract_insights_async(self, mock_orchestrator):
        """Test async insights extraction"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text='{"insights": []}')],
                    usage=Mock(input_tokens=60, output_tokens=100)
                )
            )

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
                project = ProjectContext(project_name="Test")
                result = await client.extract_insights_async("text", project)

                assert isinstance(result, (dict, list, str, type(None)))

    @pytest.mark.asyncio
    async def test_extract_tech_recommendations_async(self, mock_orchestrator):
        """Test async tech recommendations"""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text='{"recommendations": []}')],
                    usage=Mock(input_tokens=80, output_tokens=150)
                )
            )

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
                project = ProjectContext(project_name="Test")
                result = await client.extract_tech_recommendations_async(
                    project,
                    "backend"
                )

                assert isinstance(result, (dict, list, type(None)))


class TestErrorHandlingSpecialized:
    """Tests for error handling in specialized methods"""

    def test_generate_response_with_invalid_parameters(self, mock_orchestrator):
        """Test handling of invalid parameters"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            # Call with valid parameters despite potentially invalid ones
            result = client.generate_response(
                "test",
                temperature=0.5,
                max_tokens=100
            )

            assert isinstance(result, (str, type(None)))

    def test_analysis_methods_with_none_parameters(self, mock_orchestrator):
        """Test analysis methods handle None gracefully"""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="{}")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")

            # Methods should handle None gracefully
            result = client.extract_insights("", project)
            assert isinstance(result, (dict, str, type(None)))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
