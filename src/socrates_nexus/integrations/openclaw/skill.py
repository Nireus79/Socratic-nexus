"""
Socratic Nexus LLM Skill for Openclaw.

Provides multi-provider LLM support within Openclaw framework.
"""

from typing import Optional, Any, Dict
from socrates_nexus import LLMClient, AsyncLLMClient, ChatResponse, TokenUsage, LLMConfig


class NexusLLMSkill:
    """
    Openclaw Skill that provides multi-provider LLM support through Socrates Nexus.

    Enables Openclaw users to:
    - Switch between LLM providers (Claude, GPT-4, Gemini, Ollama)
    - Use automatic retry and exponential backoff
    - Track token usage and costs
    - Stream responses
    - Use fallback providers

    Example:
        >>> skill = NexusLLMSkill(provider="anthropic", model="claude-opus")
        >>> response = skill.query("What is machine learning?")
        >>> print(response.content)
        >>> print(f"Cost: ${response.usage.cost_usd}")
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-opus",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        retry_attempts: int = 3,
        cache_responses: bool = True,
        cache_ttl: int = 300,
    ):
        """
        Initialize Nexus LLM Skill for Openclaw.

        Args:
            provider: LLM provider (anthropic, openai, google, ollama)
            model: Model name (e.g., claude-opus, gpt-4, gemini-pro)
            api_key: API key for provider (uses env var if not provided)
            temperature: Sampling temperature (0-2)
            max_tokens: Max tokens in response
            retry_attempts: Number of retry attempts on failure
            cache_responses: Whether to cache responses
            cache_ttl: Cache time-to-live in seconds
        """
        self.config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_attempts=retry_attempts,
            cache_responses=cache_responses,
            cache_ttl=cache_ttl,
        )
        self.client = LLMClient(config=self.config)

    def query(self, message: str, **kwargs) -> ChatResponse:
        """
        Execute a query through the LLM.

        Args:
            message: User message/prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ChatResponse with content and usage information

        Raises:
            NexusError: If request fails after retries
        """
        return self.client.chat(message, **kwargs)

    def stream(self, message: str, on_chunk) -> ChatResponse:
        """
        Execute a streaming query.

        Args:
            message: User message/prompt
            on_chunk: Callback function called for each chunk

        Returns:
            ChatResponse with final content and usage information
        """
        return self.client.stream(message, on_chunk=on_chunk)

    def switch_provider(self, provider: str, model: str, api_key: Optional[str] = None):
        """
        Switch to a different provider.

        Args:
            provider: New provider (anthropic, openai, google, ollama)
            model: New model name
            api_key: Optional new API key
        """
        self.config.provider = provider
        self.config.model = model
        if api_key:
            self.config.api_key = api_key
        self.client = LLMClient(config=self.config)

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get cumulative usage statistics.

        Returns:
            Dictionary with token counts and costs
        """
        stats = self.client.get_usage_stats()
        return {
            "total_requests": stats.total_requests,
            "total_input_tokens": stats.total_input_tokens,
            "total_output_tokens": stats.total_output_tokens,
            "total_cost_usd": stats.total_cost_usd,
            "by_provider": stats.by_provider,
            "by_model": stats.by_model,
        }

    def add_usage_callback(self, callback):
        """
        Add callback for token usage tracking.

        Args:
            callback: Function called with TokenUsage on each request
        """
        self.client.add_usage_callback(callback)

    def __repr__(self) -> str:
        return (
            f"NexusLLMSkill("
            f"provider={self.config.provider}, "
            f"model={self.config.model})"
        )
