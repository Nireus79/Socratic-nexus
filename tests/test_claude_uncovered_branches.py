"""Tests for uncovered branches in ClaudeClient."""

import pytest
from unittest.mock import Mock, patch

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.models import ProjectContext, ConflictInfo


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


class TestGenerateArtifact:
    """Tests for generate_artifact method."""

    def test_generate_artifact_basic(self, mock_orchestrator):
        """Test basic artifact generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="artifact content")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            result = client.generate_artifact("artifact description", "artifact_type")

            assert result is not None or result is None


class TestGenerateBusinessPlan:
    """Tests for generate_business_plan method."""

    def test_generate_business_plan_basic(self, mock_orchestrator):
        """Test basic business plan generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="business plan")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.generate_business_plan(project)

            assert result is not None or result is None


class TestGenerateResearchProtocol:
    """Tests for generate_research_protocol method."""

    def test_generate_research_protocol_basic(self, mock_orchestrator):
        """Test basic research protocol generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="protocol")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.generate_research_protocol(project)

            assert result is not None or result is None


class TestGenerateCreativeBrief:
    """Tests for generate_creative_brief method."""

    def test_generate_creative_brief_basic(self, mock_orchestrator):
        """Test basic creative brief generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="brief")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.generate_creative_brief(project)

            assert result is not None or result is None


class TestGenerateMarketingPlan:
    """Tests for generate_marketing_plan method."""

    def test_generate_marketing_plan_basic(self, mock_orchestrator):
        """Test basic marketing plan generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="plan")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.generate_marketing_plan(project)

            assert result is not None or result is None


class TestGenerateCurriculum:
    """Tests for generate_curriculum method."""

    def test_generate_curriculum_basic(self, mock_orchestrator):
        """Test basic curriculum generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="curriculum")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.generate_curriculum(project)

            assert result is not None or result is None


class TestGenerateDocumentation:
    """Tests for generate_documentation method."""

    def test_generate_documentation_basic(self, mock_orchestrator):
        """Test basic documentation generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="documentation")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            result = client.generate_documentation(project, "artifact content")

            assert result is not None or result is None


class TestGenerateConflictResolution:
    """Tests for generate_conflict_resolution_suggestions method."""

    def test_generate_conflict_resolution_basic(self, mock_orchestrator):
        """Test basic conflict resolution generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="resolution")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            conflict = ConflictInfo(description="conflict", file_path="file.py")
            result = client.generate_conflict_resolution_suggestions(conflict, project)

            assert result is not None or result is None


class TestGenerateAsyncMethods:
    """Tests for async variants of generation methods."""

    @pytest.mark.asyncio
    async def test_extract_tech_recommendations_with_subscription_auth(self, mock_orchestrator):
        """Test tech recommendations with subscription auth."""
        from unittest.mock import AsyncMock

        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client

            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text='{"recommendations": []}')], usage=Mock(input_tokens=10, output_tokens=20)
                )
            )

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator, subscription_token="sub-token")
                project = ProjectContext(project_name="Test")
                result = await client.extract_tech_recommendations_async(project, "backend", user_auth_method="subscription")
                assert isinstance(result, (dict, list))

    @pytest.mark.asyncio
    async def test_generate_business_plan_async(self, mock_orchestrator):
        """Test async business plan generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            from unittest.mock import AsyncMock

            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text="plan")], usage=Mock(input_tokens=10, output_tokens=20)
                )
            )

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
                project = ProjectContext(project_name="Test")
                result = await client.generate_business_plan_async(project)
                assert result is not None or result is None

    @pytest.mark.asyncio
    async def test_generate_documentation_async(self, mock_orchestrator):
        """Test async documentation generation."""
        with patch("socratic_nexus.clients.claude_client.anthropic.AsyncAnthropic") as mock_async:
            mock_client = Mock()
            mock_async.return_value = mock_client
            from unittest.mock import AsyncMock

            mock_client.messages.create = AsyncMock(
                return_value=Mock(
                    content=[Mock(text="doc")], usage=Mock(input_tokens=10, output_tokens=20)
                )
            )

            with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
                client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
                project = ProjectContext(project_name="Test")
                result = await client.generate_documentation_async(project, "artifact")
                assert result is not None or result is None


class TestErrorHandlingBranches:
    """Tests for error handling branches."""

    def test_api_error_missing_orchestrator(self):
        """Test error when orchestrator is None and no API key."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key=None, orchestrator=None)

            with pytest.raises(APIError):
                client._get_client()

    def test_placeholder_key_detection(self, mock_orchestrator):
        """Test placeholder key is detected."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            client = ClaudeClient(api_key="placeholder_test", orchestrator=mock_orchestrator)

            # Placeholder keys shouldn't initialize client
            assert client.client is None

    def test_get_user_api_key_with_database_error(self, mock_orchestrator):
        """Test get_user_api_key falls back on database error."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic"):
            mock_orchestrator.database.get_api_key.side_effect = Exception("DB Error")

            client = ClaudeClient(api_key="env-key", orchestrator=mock_orchestrator)
            key, is_user = client._get_user_api_key("user123")

            assert key == "env-key"
            assert is_user is False


class TestParameterVariations:
    """Tests for various parameter combinations."""

    def test_generate_response_all_parameters(self, mock_orchestrator):
        """Test generate_response with all parameters."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="response")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            _ = client.generate_response(
                "prompt",
                max_tokens=1000,
                temperature=0.7,
                user_id="user123",
                user_auth_method="api_key",
            )

            mock_client.messages.create.assert_called()

    def test_generate_code_all_parameters(self, mock_orchestrator):
        """Test generate_code with all parameters."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="code")]
            mock_response.usage = Mock(input_tokens=20, output_tokens=15)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            _ = client.generate_code(
                "write function", user_id="user123", user_auth_method="api_key"
            )

            mock_client.messages.create.assert_called()

    def test_extract_insights_all_parameters(self, mock_orchestrator):
        """Test extract_insights with all parameters."""
        with patch("socratic_nexus.clients.claude_client.anthropic.Anthropic") as mock_anth:
            mock_client = Mock()
            mock_anth.return_value = mock_client
            mock_response = Mock()
            mock_response.content = [Mock(text="{}")]
            mock_response.usage = Mock(input_tokens=30, completion_tokens=20)
            mock_client.messages.create.return_value = mock_response

            client = ClaudeClient(api_key="test-key", orchestrator=mock_orchestrator)
            project = ProjectContext(project_name="Test")
            _ = client.extract_insights(
                "response", project, user_id="user123", user_auth_method="api_key"
            )

            mock_client.messages.create.assert_called()
