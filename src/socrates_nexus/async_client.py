"""Asynchronous LLM client for Socrates Nexus."""

from typing import Optional, Callable, List
from .models import ChatResponse, LLMConfig, UsageStats
from .exceptions import ConfigurationError


class AsyncLLMClient:
    """
    Asynchronous universal LLM client for production use.

    Supports all the same features as LLMClient but with async/await.
    """

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize async LLM client.

        Args:
            config: LLMConfig instance
            **kwargs: Alternative config parameters
        """
        if config is None:
            provider = kwargs.pop("provider", None)
            if not provider:
                raise ConfigurationError("Provider must be specified")
            config = LLMConfig(provider=provider, **kwargs)

        self.config = config
        self.usage_stats = UsageStats()
        self._provider = None

    async def chat(self, message: str, **kwargs) -> ChatResponse:
        """
        Async send a chat message and get response.

        Args:
            message: User message
            **kwargs: Additional arguments

        Returns:
            ChatResponse with content and usage
        """
        raise NotImplementedError("Provider not yet initialized")

    async def stream(
        self, message: str, on_chunk: Callable[[str], None], **kwargs
    ) -> ChatResponse:
        """
        Async stream a chat response with callback.

        Args:
            message: User message
            on_chunk: Async callback function for each chunk
            **kwargs: Additional arguments

        Returns:
            ChatResponse with complete content and usage
        """
        raise NotImplementedError("Provider not yet initialized")

    def get_usage_stats(self) -> UsageStats:
        """Get cumulative usage statistics."""
        return self.usage_stats
