"""
Socrates Nexus - Universal LLM client for production.

Works with Claude, GPT-4, Gemini, Llama, and any LLM with production patterns:
- Automatic retry logic with exponential backoff
- Token usage tracking and cost calculation
- Streaming support with helpers
- Async + sync APIs
- Multi-model fallback
- Type hints throughout
"""

__version__ = "0.1.0"
__author__ = "Socrates Nexus Contributors"

from .client import LLMClient
from .async_client import AsyncLLMClient
from .models import ChatResponse, TokenUsage, LLMConfig
from .exceptions import (
    LLMError,
    RateLimitError,
    InvalidAPIKeyError,
    ContextLengthExceededError,
    ModelNotFoundError,
)

__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "ChatResponse",
    "TokenUsage",
    "LLMConfig",
    "LLMError",
    "RateLimitError",
    "InvalidAPIKeyError",
    "ContextLengthExceededError",
    "ModelNotFoundError",
]
