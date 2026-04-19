"""Insight extraction and analysis for Socratic dialogue."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import logging

from .client import LLMClient
from .async_client import AsyncLLMClient
from .models import ChatResponse
from .exceptions import LLMError, InvalidRequestError

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    text: str
    category: str
    confidence: float
    source: str
    importance: str = "medium"
    related_concepts: List[str] = field(default_factory=list)
    requires_clarification: bool = False
    id: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"insight_{hash(self.text) % 10**8}"
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InsightPattern:
    pattern_type: str
    description: str
    insights_involved: List[str]
    frequency: int
    severity: str = "medium"
    recommended_action: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InsightExtractor:
    def __init__(
        self, client: Optional[LLMClient] = None, async_client: Optional[AsyncLLMClient] = None
    ) -> None:
        self._client = client
        self._async_client = async_client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient()
        return self._client

    @property
    def async_client(self) -> AsyncLLMClient:
        if self._async_client is None:
            self._async_client = AsyncLLMClient()
        return self._async_client

    def extract_insights(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        max_insights: int = 10,
        min_confidence: float = 0.5,
    ) -> List[Insight]:
        if not text or not text.strip():
            raise InvalidRequestError("Text cannot be empty")
        try:
            response = self.client.chat(f"Extract insights: {text}", system="Extract insights")
            return self._parse_insights_response(response, min_confidence)[:max_insights]
        except Exception as e:
            if isinstance(e, (InvalidRequestError, LLMError)):
                raise
            raise LLMError(f"Extraction failed: {str(e)}")

    async def aextract_insights(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        max_insights: int = 10,
        min_confidence: float = 0.5,
    ) -> List[Insight]:
        if not text or not text.strip():
            raise InvalidRequestError("Text cannot be empty")
        return []

    def _parse_insights_response(
        self, response: ChatResponse, min_confidence: float
    ) -> List[Insight]:
        try:
            content_item = response.content[0] if response.content else None
            content = content_item.text if content_item and hasattr(content_item, "text") else ""
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, ValueError, TypeError):
                return []
            if isinstance(data, dict):
                data = [data]
            insights = []
            for item in data if isinstance(data, list) else []:
                if isinstance(item, dict):
                    confidence = float(item.get("confidence", 0.5))
                    if confidence >= min_confidence:
                        insights.append(
                            Insight(
                                text=item.get("text", ""),
                                category=item.get("category", "general"),
                                confidence=confidence,
                                source=item.get("source", ""),
                            )
                        )
            return insights
        except Exception:
            return []


class InsightAnalyzer:
    def __init__(
        self, client: Optional[LLMClient] = None, async_client: Optional[AsyncLLMClient] = None
    ) -> None:
        self._client = client
        self._async_client = async_client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient()
        return self._client

    @property
    def async_client(self) -> AsyncLLMClient:
        if self._async_client is None:
            self._async_client = AsyncLLMClient()
        return self._async_client

    def analyze_insights(self, insights: List[Insight]) -> Dict[str, Any]:
        if not insights:
            return {
                "patterns": [],
                "summary": "No insights",
                "recommendations": [],
                "key_findings": [],
            }
        patterns = self._identify_patterns(insights)
        gaps = self._identify_knowledge_gaps(insights)
        return {
            "patterns": [p.to_dict() for p in patterns],
            "summary": f"Analyzed {len(insights)} insights",
            "recommendations": ["Continue learning"],
            "key_findings": self._extract_key_findings(insights),
            "knowledge_gaps": [g.to_dict() for g in gaps],
        }

    async def aanalyze_insights(self, insights: List[Insight]) -> Dict[str, Any]:
        return self.analyze_insights(insights)

    def _identify_patterns(self, insights: List[Insight]) -> List[InsightPattern]:
        patterns = []
        categories: dict[str, list[str]] = {}
        for insight in insights:
            if insight.category not in categories:
                categories[insight.category] = []
            categories[insight.category].append(insight.id)
        for category, ids in categories.items():
            if len(ids) > 1:
                patterns.append(
                    InsightPattern(
                        pattern_type="recurring_theme",
                        description=f"Multiple {category}",
                        insights_involved=ids,
                        frequency=len(ids),
                    )
                )
        return patterns

    def _identify_knowledge_gaps(self, insights: List[Insight]) -> List[InsightPattern]:
        gaps = []
        low = [i for i in insights if i.confidence < 0.6]
        if low:
            gaps.append(
                InsightPattern(
                    pattern_type="knowledge_gap",
                    description="Low confidence",
                    insights_involved=[i.id for i in low],
                    frequency=len(low),
                )
            )
        return gaps

    def _extract_key_findings(self, insights: List[Insight]) -> List[str]:
        key = sorted(
            [i for i in insights if i.confidence > 0.8], key=lambda x: x.confidence, reverse=True
        )
        return [i.text for i in key[:3]]
