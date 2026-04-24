"""Data models for Socratic Nexus LLM client library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class TokenUsage:
    """Token usage statistics for LLM API calls."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    provider: str
    model: str
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class ChatResponse:
    """Response from an LLM API call."""

    content: str
    usage: TokenUsage
    model: str
    provider: str
    stop_reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ProjectContext:
    """Context about a project for AI analysis."""

    project_name: str
    description: str = ""
    files: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictInfo:
    """Information about a code conflict."""

    description: str
    file_path: str = ""
    line_number: int = 0
    resolution_options: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["TokenUsage", "ChatResponse", "ProjectContext", "ConflictInfo"]
