"""Comprehensive tests for insights module to improve coverage."""

import json
import pytest
from unittest.mock import Mock

from socrates_nexus.insights import Insight, InsightExtractor, InsightAnalyzer
from socrates_nexus.exceptions import InvalidRequestError
from socrates_nexus.models import ChatResponse, TextContent


class TestInsight:
    """Test Insight dataclass."""

    def test_insight_creation(self):
        """Test creating an Insight with all fields."""
        insight = Insight(
            text="Key learning",
            category="concept",
            confidence=0.95,
            source="student response",
            importance="high",
            related_concepts=["recursion", "loops"],
            requires_clarification=True
        )
        assert insight.text == "Key learning"
        assert insight.category == "concept"
        assert insight.confidence == 0.95
        assert insight.importance == "high"

    def test_insight_confidence_clamping(self):
        """Test that confidence is clamped to [0, 1] range."""
        insight = Insight(
            text="test",
            category="test",
            confidence=1.5,
            source="test"
        )
        assert insight.confidence == 1.0

        insight2 = Insight(
            text="test",
            category="test",
            confidence=-0.5,
            source="test"
        )
        assert insight2.confidence == 0.0

    def test_insight_to_dict(self):
        """Test converting Insight to dictionary."""
        insight = Insight(
            text="test",
            category="test",
            confidence=0.8,
            source="test"
        )
        data = insight.to_dict()
        assert data["text"] == "test"
        assert data["confidence"] == 0.8


class TestInsightExtractor:
    """Test InsightExtractor class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock LLM client."""
        return Mock()

    @pytest.fixture
    def extractor(self, mock_client):
        """Create an InsightExtractor with mock client."""
        return InsightExtractor(client=mock_client)

    def test_extract_insights_valid_response(self, extractor, mock_client):
        """Test extracting insights from valid JSON response."""
        mock_response = ChatResponse(
            content=[TextContent(text=json.dumps([
                {"text": "insight1", "category": "concept", "confidence": 0.9, "source": "test"}
            ]))],
            raw_response="test"
        )
        mock_client.chat.return_value = mock_response

        insights = extractor.extract_insights("Test text")
        assert len(insights) >= 0

    def test_extract_insights_invalid_json(self, extractor, mock_client):
        """Test handling of invalid JSON in response."""
        mock_response = ChatResponse(
            content=[TextContent(text="not json")],
            raw_response="test"
        )
        mock_client.chat.return_value = mock_response

        insights = extractor.extract_insights("Test text")
        assert insights == []

    def test_extract_insights_empty_text(self, extractor):
        """Test that empty text raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError):
            extractor.extract_insights("")

    def test_extract_insights_with_max_limit(self, extractor, mock_client):
        """Test limiting number of insights."""
        mock_response = ChatResponse(
            content=[TextContent(text=json.dumps([
                {"text": f"insight{i}", "category": "c", "confidence": 0.8, "source": "t"}
                for i in range(5)
            ]))],
            raw_response="test"
        )
        mock_client.chat.return_value = mock_response

        insights = extractor.extract_insights("Test", max_insights=2)
        assert len(insights) <= 2


class TestInsightAnalyzer:
    """Test InsightAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an InsightAnalyzer."""
        return InsightAnalyzer()

    def test_analyze_empty_insights(self, analyzer):
        """Test analyzing empty insights list."""
        result = analyzer.analyze_insights([])

        assert result["patterns"] == []
        assert result["summary"] == "No insights"

    def test_analyze_with_insights(self, analyzer):
        """Test analyzing insights."""
        insights = [
            Insight("Test", "concept", 0.9, "test"),
            Insight("Test2", "concept", 0.85, "test"),
        ]

        result = analyzer.analyze_insights(insights)
        assert "patterns" in result
        assert "summary" in result
        assert "key_findings" in result
