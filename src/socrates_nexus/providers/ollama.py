"""Ollama local LLM provider for Socrates Nexus."""
# mypy: disable-error-code="return,call-arg"

import time
from typing import Callable

from ..models import LLMConfig, ChatResponse
from ..retry import retry_with_backoff
from ..streaming import StreamHandler, AsyncStreamHandler
from ..exceptions import (
    ProviderError,
    InvalidRequestError,
)
from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider implementation."""

    def __init__(self, config: LLMConfig):
        """
        Initialize Ollama provider.

        Args:
            config: LLM configuration (base_url defaults to http://localhost:11434)
        """
        super().__init__(config)

        # Set default base_url if not provided
        if not config.base_url:
            config.base_url = "http://localhost:11434"

        self.base_url = config.base_url
        self._client = None
        self._async_client = None

    @property
    def client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ProviderError(
                    "ollama package is required. Install with: pip install 'socrates-nexus[ollama]'"
                )
            except Exception as e:
                raise ProviderError(f"Failed to initialize Ollama client: {str(e)}")

        return self._client

    @property
    def async_client(self):
        """Lazy initialization of async Ollama client."""
        if self._async_client is None:
            try:
                import ollama

                self._async_client = ollama.AsyncClient(host=self.base_url)
            except ImportError:
                raise ProviderError(
                    "ollama package is required. Install with: pip install 'socrates-nexus[ollama]'"
                )
            except Exception as e:
                raise ProviderError(f"Failed to initialize async Ollama client: {str(e)}")

        return self._async_client

    @retry_with_backoff(
        max_attempts=3, backoff_factor=2.0, initial_delay=1.0, max_delay=32.0, jitter=True
    )
    def chat(self, message: str, **kwargs) -> ChatResponse:
        """
        Send a chat message to local Ollama model.

        Note: Ollama doesn't track token usage like cloud APIs, so counts are estimates.

        Args:
            message: User message
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Chat response with content and usage information
        """
        start_time = time.time()

        try:
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)

            # Build options for Ollama
            options = {
                "temperature": temperature,
                "top_p": top_p,
            }

            response = self.client.chat(
                model=self.config.model,
                messages=[{"role": "user", "content": message}],
                options=options,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Ollama provides eval_count (output) and prompt_eval_count (input)
            # These are approximate token counts
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)

            usage = self._create_token_usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

            # Note: Cost is 0 for local models
            self._notify_usage(usage)

            return ChatResponse(
                content=response["message"]["content"],
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=response.get("done", True) and "stop" or "continue",
            )

        except Exception as e:
            self._handle_ollama_error(e, "chat")

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
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)

            options = {
                "temperature": temperature,
                "top_p": top_p,
            }

            response = await self.async_client.chat(
                model=self.config.model,
                messages=[{"role": "user", "content": message}],
                options=options,
            )

            latency_ms = (time.time() - start_time) * 1000

            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)

            usage = self._create_token_usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=response["message"]["content"],
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=response.get("done", True) and "stop" or "continue",
            )

        except Exception as e:
            self._handle_ollama_error(e, "achat")

    def stream(self, message: str, on_chunk: Callable[[str], None], **kwargs) -> ChatResponse:
        """
        Stream chat response from local Ollama model.

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
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)

            options = {
                "temperature": temperature,
                "top_p": top_p,
            }

            prompt_tokens = 0
            completion_tokens = 0

            stream = self.client.chat(
                model=self.config.model,
                messages=[{"role": "user", "content": message}],
                options=options,
                stream=True,
            )

            for chunk in stream:
                if chunk["message"]["content"]:
                    handler.handle_chunk(chunk["message"]["content"])

                # Extract token counts from final chunk
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)

            latency_ms = (time.time() - start_time) * 1000

            usage = self._create_token_usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=handler.get_complete_response(),
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason="stop",
            )

        except Exception as e:
            self._handle_ollama_error(e, "stream")

    async def astream(
        self, message: str, on_chunk: Callable[[str], None], **kwargs
    ) -> ChatResponse:
        """
        Async version of stream.

        Args:
            message: User message
            on_chunk: Callback for each chunk
            **kwargs: Additional parameters

        Returns:
            Chat response with accumulated content and usage
        """
        start_time = time.time()
        handler = AsyncStreamHandler(on_chunk)

        try:
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)

            options = {
                "temperature": temperature,
                "top_p": top_p,
            }

            prompt_tokens = 0
            completion_tokens = 0

            stream = await self.async_client.chat(
                model=self.config.model,
                messages=[{"role": "user", "content": message}],
                options=options,
                stream=True,
            )

            async for chunk in stream:
                if chunk["message"]["content"]:
                    await handler.handle_chunk(chunk["message"]["content"])

                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)

            latency_ms = (time.time() - start_time) * 1000

            usage = self._create_token_usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=handler.get_complete_response(),
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason="stop",
            )

        except Exception as e:
            self._handle_ollama_error(e, "astream")

    def _handle_ollama_error(self, error: Exception, operation: str) -> None:
        """
        Convert Ollama errors to Nexus exceptions.

        Args:
            error: Ollama error
            operation: Operation that failed

        Raises:
            Appropriate NexusError subclass
        """
        error_msg = str(error)

        # Map Ollama errors
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            raise ProviderError(
                f"Could not connect to Ollama at {self.base_url}. Make sure Ollama is running.",
                error_code="OLLAMA_CONNECTION_ERROR",
            )
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            raise InvalidRequestError(
                f"Model {self.config.model} not found in Ollama. Run 'ollama pull {self.config.model}'",
                error_code="OLLAMA_MODEL_NOT_FOUND",
            )
        else:
            raise ProviderError(
                f"{operation} failed with Ollama: {error_msg}",
                error_code="OLLAMA_ERROR",
            )
