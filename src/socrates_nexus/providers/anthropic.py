"""Anthropic Claude provider for Socrates Nexus."""

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

from ..models import LLMConfig, ChatResponse
from ..retry import retry_with_backoff
from ..streaming import StreamHandler, AsyncStreamHandler
from ..exceptions import (
    RateLimitError,
    AuthenticationError,
    ProviderError,
    InvalidRequestError,
)
from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, config: LLMConfig):
        """
        Initialize Anthropic provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        if not config.api_key:
            raise AuthenticationError("Anthropic API key is required")

        # Lazy initialization pattern extracted from Socrates ClaudeClient (lines 32-73)
        self._client = None
        self._async_client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError as e:
                logger.error(f"anthropic package not installed: {e}")
                raise ProviderError(
                    "anthropic package is required. Install with: pip install 'socrates-nexus[anthropic]'"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}", exc_info=True)
                raise AuthenticationError(f"Failed to initialize Anthropic client: {str(e)}")

        return self._client

    @property
    def async_client(self):
        """Lazy initialization of async Anthropic client."""
        if self._async_client is None:
            try:
                import anthropic

                self._async_client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
            except ImportError as e:
                logger.error(f"anthropic package not installed: {e}")
                raise ProviderError(
                    "anthropic package is required. Install with: pip install 'socrates-nexus[anthropic]'"
                )
            except Exception as e:
                logger.error(f"Failed to initialize async Anthropic client: {e}", exc_info=True)
                raise AuthenticationError(f"Failed to initialize async Anthropic client: {str(e)}")

        return self._async_client

    @retry_with_backoff(
        max_attempts=3,
        backoff_factor=2.0,
        initial_delay=1.0,
        max_delay=32.0,
        jitter=True,
    )
    def chat(self, message: str, **kwargs) -> ChatResponse:
        """
        Send a chat message to Claude and get response.

        Uses automatic retry logic with exponential backoff.

        Args:
            message: User message
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Chat response with content and usage information
        """
        start_time = time.time()

        try:
            # Extract parameters with defaults
            max_tokens = kwargs.get("max_tokens") or self.config.max_tokens or 4096
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)

            # Build API call parameters
            api_params = {
                "model": self.config.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": message}],
            }

            # Only pass one sampling parameter (prefer temperature)
            if temperature is not None:
                api_params["temperature"] = temperature
            elif top_p is not None:
                api_params["top_p"] = top_p

            # Call Claude API
            response = self.client.messages.create(**api_params)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Extract usage and create TokenUsage
            usage = self._create_token_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
            )

            # Notify usage callbacks
            self._notify_usage(usage)

            # Create response
            chat_response = ChatResponse(
                content=response.content[0].text,
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=response.stop_reason,
                raw_response={
                    "id": response.id,
                    "type": response.type,
                    "stop_reason": response.stop_reason,
                },
            )

            return chat_response

        except Exception as e:
            self._handle_anthropic_error(e, "chat")

    async def achat(self, message: str, **kwargs) -> ChatResponse:
        """
        Async version of chat.

        Args:
            message: User message
            **kwargs: Additional parameters

        Returns:
            Chat response with content and usage information
        """
        start_time = time.time()

        try:
            max_tokens = kwargs.get("max_tokens") or self.config.max_tokens or 4096
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)

            # Build API call parameters
            api_params = {
                "model": self.config.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": message}],
            }

            # Only pass one sampling parameter (prefer temperature)
            if temperature is not None:
                api_params["temperature"] = temperature
            elif top_p is not None:
                api_params["top_p"] = top_p

            response = await self.async_client.messages.create(**api_params)

            latency_ms = (time.time() - start_time) * 1000

            usage = self._create_token_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=response.content[0].text,
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=response.stop_reason,
                raw_response={"id": response.id, "type": response.type},
            )

        except Exception as e:
            self._handle_anthropic_error(e, "achat")

    def stream(self, message: str, on_chunk: Callable[[str], None], **kwargs) -> ChatResponse:
        """
        Stream chat response from Claude.

        Args:
            message: User message
            on_chunk: Callback for each streamed chunk
            **kwargs: Additional parameters

        Returns:
            Chat response with accumulated content and usage
        """
        start_time = time.time()
        handler = StreamHandler(on_chunk)

        try:
            max_tokens = kwargs.get("max_tokens") or self.config.max_tokens or 4096
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)

            # Build API call parameters
            api_params = {
                "model": self.config.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": message}],
            }

            # Only pass one sampling parameter (prefer temperature)
            if temperature is not None:
                api_params["temperature"] = temperature
            elif top_p is not None:
                api_params["top_p"] = top_p

            with self.client.messages.stream(**api_params) as stream:
                for text in stream.text_stream:
                    handler.handle_chunk(text)

            # Get final message with usage information
            final_message = stream.get_final_message()
            latency_ms = (time.time() - start_time) * 1000

            usage = self._create_token_usage(
                input_tokens=final_message.usage.input_tokens,
                output_tokens=final_message.usage.output_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=handler.get_complete_response(),
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=final_message.stop_reason,
                raw_response={"id": final_message.id, "type": final_message.type},
            )

        except Exception as e:
            self._handle_anthropic_error(e, "stream")

    async def astream(
        self, message: str, on_chunk: Callable[[str], None], **kwargs
    ) -> ChatResponse:
        """
        Async version of stream.

        Args:
            message: User message
            on_chunk: Callback for each chunk (can be async or sync)
            **kwargs: Additional parameters

        Returns:
            Chat response with accumulated content and usage
        """
        start_time = time.time()
        handler = AsyncStreamHandler(on_chunk)

        try:
            max_tokens = kwargs.get("max_tokens") or self.config.max_tokens or 4096
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)

            # Build API call parameters
            api_params = {
                "model": self.config.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": message}],
            }

            # Only pass one sampling parameter (prefer temperature)
            if temperature is not None:
                api_params["temperature"] = temperature
            elif top_p is not None:
                api_params["top_p"] = top_p

            async with self.async_client.messages.stream(**api_params) as stream:
                async for text in stream.text_stream:
                    await handler.handle_chunk(text)

            # Get final message
            final_message = stream.get_final_message()
            latency_ms = (time.time() - start_time) * 1000

            usage = self._create_token_usage(
                input_tokens=final_message.usage.input_tokens,
                output_tokens=final_message.usage.output_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=handler.get_complete_response(),
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=final_message.stop_reason,
                raw_response={"id": final_message.id, "type": final_message.type},
            )

        except Exception as e:
            self._handle_anthropic_error(e, "astream")

    def _handle_anthropic_error(self, error: Exception, operation: str) -> None:
        """
        Convert Anthropic SDK errors to Nexus exceptions.

        Args:
            error: Anthropic SDK error
            operation: Operation that failed

        Raises:
            Appropriate NexusError subclass
        """
        error_msg = str(error)

        # Map Anthropic errors to Nexus exceptions
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            raise RateLimitError(f"Anthropic rate limit exceeded: {error_msg}")
        elif "invalid_api_key" in error_msg or "401" in error_msg:
            raise AuthenticationError(f"Invalid Anthropic API key: {error_msg}")
        elif "context_length_exceeded" in error_msg:
            from ..exceptions import ContextLengthExceededError

            raise ContextLengthExceededError(f"Context length exceeded: {error_msg}")
        elif "invalid_request_error" in error_msg or "400" in error_msg:
            raise InvalidRequestError(f"Invalid request to Anthropic: {error_msg}")
        else:
            raise ProviderError(
                f"{operation} failed with Anthropic: {error_msg}",
                error_code="ANTHROPIC_ERROR",
            )
