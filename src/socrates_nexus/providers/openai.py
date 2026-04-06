"""OpenAI GPT provider for Socrates Nexus."""
# mypy: disable-error-code="index,dict-item,return,arg-type,str-bytes-safe,assignment"

import time
from typing import Callable, List, Any

from ..models import LLMConfig, ChatResponse
from ..retry import retry_with_backoff
from ..streaming import StreamHandler, AsyncStreamHandler
from ..exceptions import (
    RateLimitError,
    AuthenticationError,
    ProviderError,
    InvalidRequestError,
    ContextLengthExceededError,
)
from ..vision import VisionMessage, VisionProcessor
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI GPT provider implementation."""

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        if not config.api_key:
            raise AuthenticationError("OpenAI API key is required")

        self._client = None
        self._async_client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ProviderError(
                    "openai package is required. Install with: pip install 'socrates-nexus[openai]'"
                )
            except Exception as e:
                raise AuthenticationError(f"Failed to initialize OpenAI client: {str(e)}")

        return self._client

    @property
    def async_client(self):
        """Lazy initialization of async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI

                self._async_client = AsyncOpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ProviderError(
                    "openai package is required. Install with: pip install 'socrates-nexus[openai]'"
                )
            except Exception as e:
                raise AuthenticationError(f"Failed to initialize async OpenAI client: {str(e)}")

        return self._async_client

    def _build_vision_content(self, vision_msg: VisionMessage) -> List[Any]:
        """
        Build OpenAI-compatible vision content blocks.

        Args:
            vision_msg: VisionMessage with text and images

        Returns:
            List of content blocks for OpenAI API
        """
        contents = []

        # Add text
        if vision_msg.text:
            contents.append({"type": "text", "text": vision_msg.text})

        # Add images
        if vision_msg.images:
            detail = "high"  # Can be made configurable
            for image_source in vision_msg.images:
                image_content = VisionProcessor.prepare_image(image_source, detail)

                if image_content.source.startswith("data:"):
                    # Base64 encoded
                    _, data = image_content.source.split(",", 1)
                    contents.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_content.media_type};base64,{data}",
                                "detail": detail,
                            },
                        }
                    )
                else:
                    # URL
                    contents.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_content.source, "detail": detail},
                        }
                    )

        return contents

    @retry_with_backoff(
        max_attempts=3, backoff_factor=2.0, initial_delay=1.0, max_delay=32.0, jitter=True
    )
    def chat(self, message: str, **kwargs) -> ChatResponse:
        """
        Send a chat message to OpenAI and get response.

        Supports both text and multimodal (image+text) messages.

        Args:
            message: User message (text only) or ignored if vision_message provided
            **kwargs: Additional parameters including:
                - temperature, max_tokens, top_p: LLM parameters
                - vision_message: VisionMessage for multimodal requests
                - image_detail: "low" or "high" for vision models

        Returns:
            Chat response with content and usage information
        """
        start_time = time.time()

        try:
            max_tokens = kwargs.get("max_tokens") or self.config.max_tokens
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)
            vision_message = kwargs.get("vision_message")

            # Build message content
            if vision_message and isinstance(vision_message, VisionMessage):
                content = self._build_vision_content(vision_message)
            else:
                content = message

            # Build parameters dict
            params = {
                "model": self.config.model,
                "temperature": temperature,
                "top_p": top_p,
                "messages": [{"role": "user", "content": content}],
            }

            if max_tokens:
                params["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**params)

            latency_ms = (time.time() - start_time) * 1000

            # OpenAI uses prompt_tokens and completion_tokens instead of input/output
            usage = self._create_token_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=response.choices[0].message.content,
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                raw_response={"id": response.id, "model": response.model},
            )

        except Exception as e:
            self._handle_openai_error(e, "chat")

    async def achat(self, message: str, **kwargs) -> ChatResponse:
        """
        Async version of chat.

        Supports both text and multimodal (image+text) messages.

        Args:
            message: User message (text only) or ignored if vision_message provided
            **kwargs: Additional parameters including vision_message and image_detail

        Returns:
            Chat response with content and usage information
        """
        start_time = time.time()

        try:
            max_tokens = kwargs.get("max_tokens") or self.config.max_tokens
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)
            vision_message = kwargs.get("vision_message")

            # Build message content
            if vision_message and isinstance(vision_message, VisionMessage):
                content = self._build_vision_content(vision_message)
            else:
                content = message

            params = {
                "model": self.config.model,
                "temperature": temperature,
                "top_p": top_p,
                "messages": [{"role": "user", "content": content}],
            }

            if max_tokens:
                params["max_tokens"] = max_tokens

            response = await self.async_client.chat.completions.create(**params)

            latency_ms = (time.time() - start_time) * 1000

            usage = self._create_token_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=response.choices[0].message.content,
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                raw_response={"id": response.id, "model": response.model},
            )

        except Exception as e:
            self._handle_openai_error(e, "achat")

    def stream(self, message: str, on_chunk: Callable[[str], None], **kwargs) -> ChatResponse:
        """
        Stream chat response from OpenAI.

        Supports both text and multimodal (image+text) messages.

        Args:
            message: User message (text only) or ignored if vision_message provided
            on_chunk: Callback for each streamed chunk
            **kwargs: Additional parameters including vision_message and image_detail

        Returns:
            Chat response with accumulated content and usage
        """
        start_time = time.time()
        handler = StreamHandler(on_chunk)

        try:
            max_tokens = kwargs.get("max_tokens") or self.config.max_tokens
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)
            vision_message = kwargs.get("vision_message")

            # Build message content
            if vision_message and isinstance(vision_message, VisionMessage):
                content = self._build_vision_content(vision_message)
            else:
                content = message

            params = {
                "model": self.config.model,
                "temperature": temperature,
                "top_p": top_p,
                "messages": [{"role": "user", "content": content}],
                "stream": True,
            }

            if max_tokens:
                params["max_tokens"] = max_tokens

            total_prompt_tokens = 0
            total_completion_tokens = 0
            finish_reason = None

            with self.client.chat.completions.create(**params) as stream:
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        handler.handle_chunk(chunk.choices[0].delta.content)

                    if chunk.usage:
                        total_prompt_tokens = chunk.usage.prompt_tokens
                        total_completion_tokens = chunk.usage.completion_tokens

                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

            latency_ms = (time.time() - start_time) * 1000

            usage = self._create_token_usage(
                input_tokens=total_prompt_tokens,
                output_tokens=total_completion_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=handler.get_complete_response(),
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=finish_reason,
            )

        except Exception as e:
            self._handle_openai_error(e, "stream")

    async def astream(
        self, message: str, on_chunk: Callable[[str], None], **kwargs
    ) -> ChatResponse:
        """
        Async version of stream.

        Supports both text and multimodal (image+text) messages.

        Args:
            message: User message (text only) or ignored if vision_message provided
            on_chunk: Callback for each chunk
            **kwargs: Additional parameters including vision_message and image_detail

        Returns:
            Chat response with accumulated content and usage
        """
        start_time = time.time()
        handler = AsyncStreamHandler(on_chunk)

        try:
            max_tokens = kwargs.get("max_tokens") or self.config.max_tokens
            temperature = kwargs.get("temperature", self.config.temperature)
            top_p = kwargs.get("top_p", self.config.top_p)
            vision_message = kwargs.get("vision_message")

            # Build message content
            if vision_message and isinstance(vision_message, VisionMessage):
                content = self._build_vision_content(vision_message)
            else:
                content = message

            params = {
                "model": self.config.model,
                "temperature": temperature,
                "top_p": top_p,
                "messages": [{"role": "user", "content": content}],
                "stream": True,
            }

            if max_tokens:
                params["max_tokens"] = max_tokens

            total_prompt_tokens = 0
            total_completion_tokens = 0
            finish_reason = None

            async with self.async_client.chat.completions.create(**params) as stream:
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        await handler.handle_chunk(chunk.choices[0].delta.content)

                    if chunk.usage:
                        total_prompt_tokens = chunk.usage.prompt_tokens
                        total_completion_tokens = chunk.usage.completion_tokens

                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

            latency_ms = (time.time() - start_time) * 1000

            usage = self._create_token_usage(
                input_tokens=total_prompt_tokens,
                output_tokens=total_completion_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=handler.get_complete_response(),
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=finish_reason,
            )

        except Exception as e:
            self._handle_openai_error(e, "astream")

    def _handle_openai_error(self, error: Exception, operation: str) -> None:
        """
        Convert OpenAI SDK errors to Nexus exceptions.

        Args:
            error: OpenAI SDK error
            operation: Operation that failed

        Raises:
            Appropriate NexusError subclass
        """
        error_msg = str(error)

        # Map OpenAI errors
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            raise RateLimitError(f"OpenAI rate limit exceeded: {error_msg}")
        elif "invalid_api_key" in error_msg or "401" in error_msg:
            raise AuthenticationError(f"Invalid OpenAI API key: {error_msg}")
        elif "context_length_exceeded" in error_msg or "context_length" in error_msg.lower():
            raise ContextLengthExceededError(f"Context length exceeded: {error_msg}")
        elif "invalid_request" in error_msg or "400" in error_msg:
            raise InvalidRequestError(f"Invalid request to OpenAI: {error_msg}")
        else:
            raise ProviderError(
                f"{operation} failed with OpenAI: {error_msg}",
                error_code="OPENAI_ERROR",
            )
