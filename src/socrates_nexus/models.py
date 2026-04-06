"""Data models for Socrates Nexus."""

from typing import Optional, Dict, Any, List, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime

# Provider pricing: cost per 1,000,000 tokens (millions)
PROVIDER_PRICING = {
    "anthropic": {
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
    "openai": {
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    },
    "google": {
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-pro": {"input": 0.50, "output": 1.50},
    },
    "ollama": {
        # Local models are free
        "llama2": {"input": 0.0, "output": 0.0},
        "mistral": {"input": 0.0, "output": 0.0},
        "neural-chat": {"input": 0.0, "output": 0.0},
        "orca-mini": {"input": 0.0, "output": 0.0},
    },
}


@dataclass
class TokenUsage:
    """Token usage tracking across all providers."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float = 0.0
    provider: str = ""
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0  # Request latency in milliseconds

    @property
    def total_cost(self) -> float:
        """Get total cost in USD."""
        return self.cost_usd


@dataclass
class TextContent:
    """Text content block for multimodal messages."""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageContent:
    """Image content for vision models."""

    type: Literal["image"] = "image"
    source: Union[str, bytes] = ""  # URL, file path, or base64
    media_type: Optional[str] = None  # "image/jpeg", "image/png", etc.
    detail: Optional[str] = None  # "low", "high" for OpenAI


@dataclass
class FunctionCall:
    """Function call details."""

    name: str
    arguments: str  # JSON string of arguments


@dataclass
class ToolCall:
    """Tool call made by the LLM."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall = field(default_factory=lambda: FunctionCall("", ""))


@dataclass
class Function:
    """Function definition for LLM tool use."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format


@dataclass
class Tool:
    """Tool definition (provider-agnostic)."""

    type: Literal["function"] = "function"
    function: Function = field(default_factory=lambda: Function("", "", {}))


@dataclass
class ChatResponse:
    """Unified chat response from any LLM provider."""

    content: str
    provider: str
    model: str
    usage: TokenUsage
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None  # For function calling
    content_blocks: Optional[List[Union[TextContent, ImageContent]]] = None  # For multimodal

    def __str__(self) -> str:
        return self.content


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: str  # "anthropic", "openai", "google", "ollama", "huggingface"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    timeout: int = 30
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    cache_responses: bool = True
    cache_ttl: int = 300  # 5 minutes
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageStats:
    """Cumulative usage statistics."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    by_provider: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_usage(self, usage: TokenUsage) -> None:
        """Add usage stats."""
        self.total_requests += 1
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cost_usd += usage.cost_usd

        # Track by provider
        if usage.provider not in self.by_provider:
            self.by_provider[usage.provider] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        self.by_provider[usage.provider]["requests"] += 1
        self.by_provider[usage.provider]["input_tokens"] += usage.input_tokens
        self.by_provider[usage.provider]["output_tokens"] += usage.output_tokens
        self.by_provider[usage.provider]["cost_usd"] += usage.cost_usd

        # Track by model
        if usage.model not in self.by_model:
            self.by_model[usage.model] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        self.by_model[usage.model]["requests"] += 1
        self.by_model[usage.model]["input_tokens"] += usage.input_tokens
        self.by_model[usage.model]["output_tokens"] += usage.output_tokens
        self.by_model[usage.model]["cost_usd"] += usage.cost_usd
