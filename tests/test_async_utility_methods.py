"""Comprehensive tests for untested async utility methods in ClaudeClient.

Tests for:
- detect_conflicts_async()
- analyze_context_async()
- extract_tech_recommendations_async()
- evaluate_quality_async()
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator."""
    orch = Mock()
    orch.config = Mock()
    orch.config.claude_model = "claude-3-sonnet-20240229"
    orch.event_emitter = Mock()
    orch.database = Mock()
    orch.database.get_api_key.return_value = None
    return orch


def create_mock_async_response(
    text="response", input_tokens=10, output_tokens=20
):
    """Helper to create mock async API response."""
    response = Mock()
    response.content = [Mock(text=text)]
    response.usage = Mock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


class TestDetectConflictsAsync:
    """Tests for detect_conflicts_async method."""

    @pytest.mark.asyncio
    async def test_detect_conflicts_async_basic(self, mock_orchestrator):
        """Test basic conflict detection."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("[]")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            requirements = [
                "High performance required",
                "Low latency critical",
            ]
            result = await client.detect_conflicts_async(requirements)

            assert isinstance(result, (list, dict))

    @pytest.mark.asyncio
    async def test_detect_conflicts_async_with_conflicts(self, mock_orchestrator):
        """Test conflict detection with actual conflicts."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            response_text = """[
                {
                    "requirement_ids": ["req1", "req2"],
                    "type": "technical",
                    "severity": "high",
                    "description": "High performance conflicts with low resource usage"
                }
            ]"""
            mock_client.messages.create.return_value = (
                create_mock_async_response(response_text)
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            requirements = [
                "High performance required",
                "Minimal resource footprint",
            ]
            result = await client.detect_conflicts_async(requirements)

            assert isinstance(result, (list, dict))

    @pytest.mark.asyncio
    async def test_detect_conflicts_async_empty_requirements(self, mock_orchestrator):
        """Test with empty requirements list."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("[]")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = await client.detect_conflicts_async([])

            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_conflicts_async_error_handling(self, mock_orchestrator):
        """Test error handling in conflict detection."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("API Error")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = await client.detect_conflicts_async(["req1"])

            assert isinstance(result, list)
            assert result == []

    @pytest.mark.asyncio
    async def test_detect_conflicts_async_with_api_key_auth(self, mock_orchestrator):
        """Test with api_key authentication method."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("[]")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = await client.detect_conflicts_async(
                ["req1"], user_auth_method="api_key"
            )

            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_conflicts_async_complex_requirements(self, mock_orchestrator):
        """Test with complex requirement objects."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("[]")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            requirements = [
                {"id": "req1", "description": "High performance", "priority": "high"},
                {"id": "req2", "description": "Low latency", "priority": "critical"},
            ]
            result = await client.detect_conflicts_async(requirements)

            assert isinstance(result, list)


class TestAnalyzeContextAsync:
    """Tests for analyze_context_async method."""

    @pytest.mark.asyncio
    async def test_analyze_context_async_basic(self, mock_orchestrator):
        """Test basic context analysis."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("Analysis text")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="TestProject")
            result = await client.analyze_context_async(project)

            assert isinstance(result, str)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_analyze_context_async_full_context(self, mock_orchestrator):
        """Test with fully populated ProjectContext."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("Comprehensive analysis")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(
                project_name="FullProject",
                phase="implementation",
                goals=["goal1", "goal2"],
                tech_stack=["Python", "FastAPI"],
                status="active",
                progress=75,
            )
            result = await client.analyze_context_async(project)

            assert isinstance(result, str)
            assert "analysis" in result.lower() or len(result) > 0

    @pytest.mark.asyncio
    async def test_analyze_context_async_minimal_context(self, mock_orchestrator):
        """Test with minimal ProjectContext."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("Analysis")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Minimal")
            result = await client.analyze_context_async(project)

            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_analyze_context_async_error_handling(self, mock_orchestrator):
        """Test error handling in context analysis."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("API Error")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = await client.analyze_context_async(project)

            assert isinstance(result, str)
            assert result == ""

    @pytest.mark.asyncio
    async def test_analyze_context_async_with_long_response(self, mock_orchestrator):
        """Test with long response text."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            long_text = "Analysis " * 100
            mock_client.messages.create.return_value = (
                create_mock_async_response(long_text)
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = await client.analyze_context_async(project)

            assert isinstance(result, str)
            assert len(result) > 50

    @pytest.mark.asyncio
    async def test_analyze_context_async_various_phases(self, mock_orchestrator):
        """Test with different project phases."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("Analysis")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            phases = ["planning", "design", "implementation", "testing", "deployment"]
            for phase in phases:
                project = ProjectContext(project_name="Test", phase=phase)
                result = await client.analyze_context_async(project)
                assert isinstance(result, str)


class TestExtractTechRecommendationsAsync:
    """Tests for extract_tech_recommendations_async method."""

    @pytest.mark.asyncio
    async def test_extract_tech_recommendations_async_basic(self, mock_orchestrator):
        """Test basic tech recommendations extraction."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            response_text = '{"recommendations": [{"tech": "Python", "score": 9}]}'
            mock_client.messages.create.return_value = (
                create_mock_async_response(response_text)
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(
                project_name="Test", tech_stack=["JavaScript"]
            )
            result = await client.extract_tech_recommendations_async(
                project, "database"
            )

            assert isinstance(result, (dict, list))

    @pytest.mark.asyncio
    async def test_extract_tech_recommendations_async_for_database(
        self, mock_orchestrator
    ):
        """Test tech recommendations for database query."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            response_text = """{
                "recommendations": [
                    {"technology": "PostgreSQL", "pros": ["ACID"], "cons": ["complexity"]},
                    {"technology": "MongoDB", "pros": ["Scalability"], "cons": ["transactions"]}
                ]
            }"""
            mock_client.messages.create.return_value = (
                create_mock_async_response(response_text)
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="DataApp")
            result = await client.extract_tech_recommendations_async(project, "database")

            assert isinstance(result, (dict, list))

    @pytest.mark.asyncio
    async def test_extract_tech_recommendations_async_for_framework(
        self, mock_orchestrator
    ):
        """Test tech recommendations for framework query."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            response_text = (
                '{"recommendations": [{"framework": "FastAPI", "rating": 9}]}'
            )
            mock_client.messages.create.return_value = (
                create_mock_async_response(response_text)
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="API")
            result = await client.extract_tech_recommendations_async(
                project, "web framework"
            )

            assert isinstance(result, (dict, list))

    @pytest.mark.asyncio
    async def test_extract_tech_recommendations_async_error_handling(
        self, mock_orchestrator
    ):
        """Test error handling in tech recommendations."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("API Error")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = await client.extract_tech_recommendations_async(project, "query")

            assert isinstance(result, dict)
            assert result == {}

    @pytest.mark.asyncio
    async def test_extract_tech_recommendations_async_invalid_json(
        self, mock_orchestrator
    ):
        """Test handling of invalid JSON response."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("not valid json")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = await client.extract_tech_recommendations_async(project, "query")

            assert isinstance(result, (dict, list, type(None)))

    @pytest.mark.asyncio
    async def test_extract_tech_recommendations_async_with_tech_stack(
        self, mock_orchestrator
    ):
        """Test tech recommendations with project tech stack."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response('{"recommendations": []}')
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(
                project_name="Test",
                tech_stack=["Python", "FastAPI"],
            )
            result = await client.extract_tech_recommendations_async(
                project, "backend"
            )

            assert isinstance(result, (dict, list))

    @pytest.mark.asyncio
    async def test_extract_tech_recommendations_async_empty_query(self, mock_orchestrator):
        """Test with empty query string."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response('{"recommendations": []}')
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = await client.extract_tech_recommendations_async(project, "")

            assert isinstance(result, (dict, list))


class TestEvaluateQualityAsync:
    """Tests for evaluate_quality_async method."""

    @pytest.mark.asyncio
    async def test_evaluate_quality_async_basic(self, mock_orchestrator):
        """Test basic quality evaluation."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            response_text = '{"score": 8, "feedback": "Good code"}'
            mock_client.messages.create.return_value = (
                create_mock_async_response(response_text)
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            code = "def hello():\n    print('hello')"
            result = await client.evaluate_quality_async(code)

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_evaluate_quality_async_code_content(self, mock_orchestrator):
        """Test quality evaluation for code."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            response_text = """{
                "score": 7,
                "quality": "good",
                "feedback": "Well structured",
                "improvements": ["Add docstrings", "Add type hints"]
            }"""
            mock_client.messages.create.return_value = (
                create_mock_async_response(response_text)
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            code = "def add(a, b):\n    return a + b"
            result = await client.evaluate_quality_async(code, content_type="code")

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_evaluate_quality_async_documentation_content(
        self, mock_orchestrator
    ):
        """Test quality evaluation for documentation."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response('{"score": 9, "feedback": "Excellent docs"}')
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            docs = "# API Documentation\n\nThis API provides..."
            result = await client.evaluate_quality_async(
                docs, content_type="documentation"
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_evaluate_quality_async_error_handling(self, mock_orchestrator):
        """Test error handling in quality evaluation."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.side_effect = Exception("API Error")

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = await client.evaluate_quality_async("content")

            assert isinstance(result, dict)
            assert "score" in result or result.get("score") == 0

    @pytest.mark.asyncio
    async def test_evaluate_quality_async_empty_content(self, mock_orchestrator):
        """Test with empty content."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response('{"score": 0, "feedback": "Empty content"}')
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = await client.evaluate_quality_async("")

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_evaluate_quality_async_large_content(self, mock_orchestrator):
        """Test with large content."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response('{"score": 7, "feedback": "Long code"}')
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            large_code = "def func():\n    " + "pass\n    " * 100
            result = await client.evaluate_quality_async(large_code)

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_evaluate_quality_async_invalid_json_response(
        self, mock_orchestrator
    ):
        """Test handling of invalid JSON in response."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("not valid json")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = await client.evaluate_quality_async("code")

            assert isinstance(result, (dict, type(None)))

    @pytest.mark.asyncio
    async def test_evaluate_quality_async_various_content_types(
        self, mock_orchestrator
    ):
        """Test with various content types."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response('{"score": 8}')
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            content_types = ["code", "documentation", "design", "architecture"]
            for content_type in content_types:
                result = await client.evaluate_quality_async(
                    "sample content", content_type=content_type
                )
                assert isinstance(result, dict)


class TestAsyncMethodAuthFlow:
    """Tests for authentication flow in async methods."""

    @pytest.mark.asyncio
    async def test_detect_conflicts_with_subscription_auth(self, mock_orchestrator):
        """Test conflict detection with subscription auth."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("[]")
            )

            client = ClaudeClient(
                api_key="test-key",
                orchestrator=mock_orchestrator,
                subscription_token="sub-token",
            )
            result = await client.detect_conflicts_async(
                ["req1"], user_auth_method="subscription"
            )

            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_analyze_context_with_fallback_auth(self, mock_orchestrator):
        """Test context analysis with auth fallback."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("Analysis")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = await client.analyze_context_async(project)

            assert isinstance(result, str)


class TestAsyncMethodConcurrency:
    """Tests for concurrent execution of async methods."""

    @pytest.mark.asyncio
    async def test_concurrent_conflict_detection(self, mock_orchestrator):
        """Test concurrent conflict detection calls."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("[]")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            tasks = [
                client.detect_conflicts_async(["req1"]),
                client.detect_conflicts_async(["req2"]),
                client.detect_conflicts_async(["req3"]),
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all(isinstance(r, (list, dict)) for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_context_analysis(self, mock_orchestrator):
        """Test concurrent context analysis calls."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response("Analysis")
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)

            projects = [
                ProjectContext(project_name="Project1"),
                ProjectContext(project_name="Project2"),
                ProjectContext(project_name="Project3"),
            ]

            tasks = [
                client.analyze_context_async(p) for p in projects
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_mixed_async_method_calls(self, mock_orchestrator):
        """Test concurrent calls to different async methods."""
        with patch(
            "socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic"
        ) as mock_anth:
            mock_client = AsyncMock()
            mock_anth.return_value = mock_client
            mock_client.messages.create.return_value = (
                create_mock_async_response('{"test": "data"}')
            )

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")

            tasks = [
                client.detect_conflicts_async(["req1"]),
                client.analyze_context_async(project),
                client.extract_tech_recommendations_async(project, "database"),
                client.evaluate_quality_async("code"),
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 4
            assert isinstance(results[0], (list, dict))
            assert isinstance(results[1], str)
            assert isinstance(results[2], (list, dict))
            assert isinstance(results[3], dict)
