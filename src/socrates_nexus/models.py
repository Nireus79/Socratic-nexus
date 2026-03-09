"""Data models for Socrates Nexus."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime


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

    @property
    def total_cost(self) -> float:
        """Get total cost in USD."""
        return self.cost_usd


@dataclass
class ChatResponse:
    """Unified chat response from any LLM provider."""

    content: str
    provider: str
    model: str
    usage: TokenUsage
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

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
