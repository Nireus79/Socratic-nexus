"""
Socratic Nexus - Claude API Client Library

Extracted from Socrates v1.3.3
"""

from __future__ import annotations

from .clients.claude_client import ClaudeClient
from .models import TokenUsage, ChatResponse, ProjectContext, ConflictInfo
from .exceptions import APIError
from .events import EventType

__version__ = "0.4.0"
__all__ = [
    "ClaudeClient",
    "TokenUsage",
    "ChatResponse",
    "ProjectContext",
    "ConflictInfo",
    "APIError",
    "EventType",
]

# Optional client imports with graceful fallback
try:
    from .clients.openai_client import OpenAIClient  # noqa: F401
    __all__.append("OpenAIClient")
except ImportError:
    pass

try:
    from .clients.google_client import GoogleClient  # noqa: F401
    __all__.append("GoogleClient")
except ImportError:
    pass

try:
    from .clients.ollama_client import OllamaClient  # noqa: F401
    __all__.append("OllamaClient")
except ImportError:
    pass
