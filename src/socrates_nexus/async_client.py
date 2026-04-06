"""Asynchronous LLM client for Socrates Nexus."""

import hashlib
from typing import Optional, Callable

from .models import ChatResponse, LLMConfig, UsageStats, TokenUsage
from .exceptions import ConfigurationError, ProviderError
from .providers import (
    BaseProvider,
    AnthropicProvider,
    OpenAIProvider,
    GoogleProvider,
    OllamaProvider,
)
from .utils.cache import ResponseCache


class AsyncLLMClient:
    """
    Asynchronous universal LLM client for production use.

    Supports all the same features as LLMClient but with async/await.
    - Automatic retry logic with exponential backoff
    - Token usage tracking and cost calculation
    - Async streaming support
    - Optional response caching with TTL
    """

    PROVIDER_MAP = {
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,  # Alias
        "openai": OpenAIProvider,
        "gpt": OpenAIProvider,  # Alias
        "google": GoogleProvider,
        "gemini": GoogleProvider,  # Alias
        "ollama": OllamaProvider,
        "local": OllamaProvider,  # Alias
    }

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        """
        Initialize async LLM client.

        Args:
            config: LLMConfig instance
            **kwargs: Alternative config parameters (provider, model, api_key, etc.)

        Raises:
            ConfigurationError: If provider is not specified
        """
        if config is None:
            provider = kwargs.pop("provider", None)
            if not provider:
                raise ConfigurationError(
                    "Provider must be specified. Supported: anthropic, openai, google, ollama"
                )
            config = LLMConfig(provider=provider, **kwargs)

        self.config = config
        self.usage_stats = UsageStats()
        self._provider: Optional[BaseProvider] = None
        self._cache: Optional[ResponseCache] = None

        # Initialize cache if enabled
        if config.cache_responses:
            self._cache = ResponseCache(ttl_minutes=int(config.cache_ttl / 60))

    @property
    def provider(self) -> BaseProvider:
        """Lazy initialization of provider."""
        if self._provider is None:
            self._provider = self._create_provider()
            # Setup usage callback for tracking
            self._provider.add_usage_callback(self._track_usage)

        return self._provider

    def _create_provider(self) -> BaseProvider:
        """
        Create provider instance based on config.

        Returns:
            Initialized provider instance

        Raises:
            ProviderError: If provider is not supported
        """
        provider_name = self.config.provider.lower()

        if provider_name not in self.PROVIDER_MAP:
            raise ProviderError(
                f"Unsupported provider: {self.config.provider}. "
                f"Supported providers: {', '.join(self.PROVIDER_MAP.keys())}"
            )

        provider_class = self.PROVIDER_MAP[provider_name]
        return provider_class(self.config)  # type: ignore[abstract]

    def _track_usage(self, usage: TokenUsage) -> None:
        """
        Track token usage for statistics.

        Args:
            usage: Token usage information
        """
        self.usage_stats.add_usage(usage)

    def _get_cache_key(self, message: str) -> str:
        """
        Generate cache key for message.

        Args:
            message: Message to generate key for

        Returns:
            SHA256 hex digest
        """
        return hashlib.sha256(message.encode()).hexdigest()

    async def chat(self, message: str, **kwargs) -> ChatResponse:
        """
        Async send a chat message and get response.

        Automatically uses caching if enabled in config.

        Args:
            message: User message
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            ChatResponse with content and usage
        """
        # Check cache
        if self._cache:
            cache_key = self._get_cache_key(message)
            cached_response = self._cache.get(cache_key)
            if cached_response:
                return cached_response

        # Call provider
        response = await self.provider.achat(message, **kwargs)

        # Store in cache
        if self._cache:
            cache_key = self._get_cache_key(message)
            self._cache.set(cache_key, response)

        return response

    async def stream(self, message: str, on_chunk: Callable[[str], None], **kwargs) -> ChatResponse:
        """
        Async stream a chat response with callback.

        Note: Streaming responses are not cached.

        Args:
            message: User message
            on_chunk: Callback function for each chunk (can be async or sync)
            **kwargs: Additional arguments

        Returns:
            ChatResponse with complete content and usage
        """
        return await self.provider.astream(message, on_chunk, **kwargs)

    def get_usage_stats(self) -> UsageStats:
        """
        Get cumulative usage statistics.

        Returns:
            UsageStats with total and per-provider/model breakdowns
        """
        return self.usage_stats

    def add_usage_callback(self, callback: Callable[[TokenUsage], None]) -> None:
        """
        Add callback for token usage tracking.

        Useful for logging, monitoring, or external tracking systems.

        Args:
            callback: Function to call with TokenUsage
        """
        self.provider.add_usage_callback(callback)
