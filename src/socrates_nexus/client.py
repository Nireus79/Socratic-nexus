"""Synchronous LLM client for Socrates Nexus."""

from typing import Optional, Callable, List
from .models import ChatResponse, LLMConfig, UsageStats
from .exceptions import ConfigurationError


class LLMClient:
    """
    Universal LLM client for production use.

    Supports Claude, GPT-4, Gemini, Llama, and any LLM with production patterns:
    - Automatic retry logic with exponential backoff
    - Token usage tracking and cost calculation
    - Streaming support with helpers
    - Type hints throughout
    """

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize LLM client.

        Args:
            config: LLMConfig instance
            **kwargs: Alternative config parameters
        """
        if config is None:
            # Create config from kwargs
            provider = kwargs.pop("provider", None)
            if not provider:
                raise ConfigurationError("Provider must be specified")
            config = LLMConfig(provider=provider, **kwargs)

        self.config = config
        self.usage_stats = UsageStats()
        self._provider = None

    def chat(self, message: str, **kwargs) -> ChatResponse:
        """
        Send a chat message and get response.

        Args:
            message: User message
            **kwargs: Additional arguments (model, temperature, etc.)

        Returns:
            ChatResponse with content and usage
        """
        raise NotImplementedError("Provider not yet initialized")

    def stream(
        self, message: str, on_chunk: Callable[[str], None], **kwargs
    ) -> ChatResponse:
        """
        Stream a chat response with callback.

        Args:
            message: User message
            on_chunk: Callback function for each chunk
            **kwargs: Additional arguments

        Returns:
            ChatResponse with complete content and usage
        """
        raise NotImplementedError("Provider not yet initialized")

    def get_usage_stats(self) -> UsageStats:
        """Get cumulative usage statistics."""
        return self.usage_stats
