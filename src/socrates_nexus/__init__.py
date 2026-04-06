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
from .documentation import (
    APIDocumentation,
    APIDocumentationGenerator,
    EndpointDoc,
    ParameterDoc,
)
from .performance import (
    CostOptimizer,
    InferenceOptimizer,
    LatencyOptimizer,
    ModelOptimizationConfig,
    OptimizationResult,
    PerformanceMetrics,
)
from .deduplication import (
    RequestDeduplicator,
    RequestBatcher,
    CachedResponse,
)
from .models import ChatResponse, TokenUsage, LLMConfig, ImageContent, TextContent
from .exceptions import (
    LLMError,
    NexusError,
    RateLimitError,
    AuthenticationError,
    InvalidAPIKeyError,
    TimeoutError,
    ContextLengthExceededError,
    ModelNotFoundError,
    ProviderError,
)
from .vision import VisionMessage, VisionProcessor, VisionCapabilities

__all__ = [
    "LLMClient",
    "AsyncLLMClient",
    "ChatResponse",
    "TokenUsage",
    "LLMConfig",
    "ImageContent",
    "TextContent",
    # Vision support
    "VisionMessage",
    "VisionProcessor",
    "VisionCapabilities",
    # Documentation
    "APIDocumentation",
    "APIDocumentationGenerator",
    "EndpointDoc",
    "ParameterDoc",
    # Performance Optimization
    "InferenceOptimizer",
    "LatencyOptimizer",
    "CostOptimizer",
    "ModelOptimizationConfig",
    "PerformanceMetrics",
    "OptimizationResult",
    # Request Deduplication & Batching
    "RequestDeduplicator",
    "RequestBatcher",
    "CachedResponse",
    # Exceptions
    "LLMError",
    "NexusError",
    "RateLimitError",
    "AuthenticationError",
    "InvalidAPIKeyError",
    "TimeoutError",
    "ContextLengthExceededError",
    "ModelNotFoundError",
    "ProviderError",
]
