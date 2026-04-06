"""Google Gemini provider for Socrates Nexus."""
# mypy: disable-error-code="arg-type,return,str-bytes-safe,assignment"

import time
from typing import Callable, Any, List
from io import BytesIO

from ..models import LLMConfig, ChatResponse
from ..retry import retry_with_backoff
from ..streaming import StreamHandler, AsyncStreamHandler
from ..exceptions import (
    RateLimitError,
    AuthenticationError,
    ProviderError,
    InvalidRequestError,
)
from ..vision import VisionMessage
from ..utils.images import is_image_url, is_image_path, load_image_from_url
from .base import BaseProvider


class GoogleProvider(BaseProvider):
    """Google Gemini provider implementation."""

    def __init__(self, config: LLMConfig):
        """
        Initialize Google provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)

        if not config.api_key:
            raise AuthenticationError("Google API key is required")

        self._model = None

    @property
    def model(self):
        """Lazy initialization of Google model."""
        if self._model is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.config.api_key)
                self._model = genai.GenerativeModel(self.config.model)
            except ImportError:
                raise ProviderError(
                    "google-generativeai package is required. Install with: pip install 'socrates-nexus[google]'"
                )
            except Exception as e:
                raise AuthenticationError(f"Failed to initialize Google model: {str(e)}")

        return self._model

    def _build_vision_content(self, vision_msg: VisionMessage) -> List[Any]:
        """
        Build Google-compatible vision content.

        Args:
            vision_msg: VisionMessage with text and images

        Returns:
            List of content items for Gemini API
        """
        try:
            from PIL import Image
        except ImportError:
            raise ProviderError(
                "Pillow is required for vision support. Install with: pip install 'socrates-nexus[vision]'"
            )

        contents = []

        # Add text
        if vision_msg.text:
            contents.append(vision_msg.text)

        # Add images
        if vision_msg.images:
            for image_source in vision_msg.images:
                try:
                    if is_image_url(image_source):
                        # Download and open URL image
                        image_bytes = load_image_from_url(image_source)
                        img = Image.open(BytesIO(image_bytes))
                    elif is_image_path(image_source):
                        # Open file path image
                        img = Image.open(image_source)
                    elif isinstance(image_source, bytes):
                        # Open bytes image
                        img = Image.open(BytesIO(image_source))
                    else:
                        raise ValueError(f"Invalid image source: {image_source}")

                    contents.append(img)

                except Exception as e:
                    raise ProviderError(f"Failed to process image '{image_source}': {str(e)}")

        return contents

    @retry_with_backoff(
        max_attempts=3, backoff_factor=2.0, initial_delay=1.0, max_delay=32.0, jitter=True
    )
    def chat(self, message: str, **kwargs) -> ChatResponse:
        """
        Send a chat message to Gemini and get response.

        Supports both text and multimodal (image+text) messages.

        Args:
            message: User message (text only) or ignored if vision_message provided
            **kwargs: Additional parameters including:
                - temperature, max_tokens, top_p: LLM parameters
                - vision_message: VisionMessage for multimodal requests

        Returns:
            Chat response with content and usage information
        """
        start_time = time.time()

        try:
            generation_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }

            if kwargs.get("max_tokens") or self.config.max_tokens:
                generation_config["max_output_tokens"] = (
                    kwargs.get("max_tokens") or self.config.max_tokens
                )

            # Build message content
            vision_message = kwargs.get("vision_message")
            if vision_message and isinstance(vision_message, VisionMessage):
                content = self._build_vision_content(vision_message)
            else:
                content = message

            response = self.model.generate_content(
                content,
                generation_config=generation_config,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage info from response
            prompt_tokens = (
                response.usage_metadata.prompt_character_count if response.usage_metadata else 0
            )
            completion_tokens = (
                response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            )

            usage = self._create_token_usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=response.text,
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=(
                    response.candidates[0].finish_reason.name if response.candidates else None
                ),
            )

        except Exception as e:
            self._handle_google_error(e, "chat")

    async def achat(self, message: str, **kwargs) -> ChatResponse:
        """
        Async version of chat.

        Supports both text and multimodal (image+text) messages.

        Args:
            message: User message (text only) or ignored if vision_message provided
            **kwargs: Additional parameters including vision_message

        Returns:
            Chat response with content and usage information
        """
        start_time = time.time()

        try:
            generation_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }

            if kwargs.get("max_tokens") or self.config.max_tokens:
                generation_config["max_output_tokens"] = (
                    kwargs.get("max_tokens") or self.config.max_tokens
                )

            # Build message content
            vision_message = kwargs.get("vision_message")
            if vision_message and isinstance(vision_message, VisionMessage):
                content = self._build_vision_content(vision_message)
            else:
                content = message

            response = await self.model.generate_content_async(
                content,
                generation_config=generation_config,
            )

            latency_ms = (time.time() - start_time) * 1000

            prompt_tokens = (
                response.usage_metadata.prompt_character_count if response.usage_metadata else 0
            )
            completion_tokens = (
                response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            )

            usage = self._create_token_usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

            self._notify_usage(usage)

            return ChatResponse(
                content=response.text,
                provider=self.config.provider,
                model=self.config.model,
                usage=usage,
                finish_reason=(
                    response.candidates[0].finish_reason.name if response.candidates else None
                ),
            )

        except Exception as e:
            self._handle_google_error(e, "achat")

    def stream(self, message: str, on_chunk: Callable[[str], None], **kwargs) -> ChatResponse:
        """
        Stream chat response from Gemini.

        Supports both text and multimodal (image+text) messages.

        Args:
            message: User message (text only) or ignored if vision_message provided
            on_chunk: Callback for each streamed chunk
            **kwargs: Additional parameters including vision_message

        Returns:
            Chat response with accumulated content and usage
        """
        start_time = time.time()
        handler = StreamHandler(on_chunk)

        try:
            generation_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }

            if kwargs.get("max_tokens") or self.config.max_tokens:
                generation_config["max_output_tokens"] = (
                    kwargs.get("max_tokens") or self.config.max_tokens
                )

            # Build message content
            vision_message = kwargs.get("vision_message")
            if vision_message and isinstance(vision_message, VisionMessage):
                content = self._build_vision_content(vision_message)
            else:
                content = message

            response = self.model.generate_content(
                content,
                generation_config=generation_config,
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    handler.handle_chunk(chunk.text)

            latency_ms = (time.time() - start_time) * 1000

            prompt_tokens = (
                response.usage_metadata.prompt_character_count if response.usage_metadata else 0
            )
            completion_tokens = (
                response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            )

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
                finish_reason=(
                    response.candidates[0].finish_reason.name if response.candidates else None
                ),
            )

        except Exception as e:
            self._handle_google_error(e, "stream")

    async def astream(
        self, message: str, on_chunk: Callable[[str], None], **kwargs
    ) -> ChatResponse:
        """
        Async version of stream.

        Supports both text and multimodal (image+text) messages.

        Args:
            message: User message (text only) or ignored if vision_message provided
            on_chunk: Callback for each chunk
            **kwargs: Additional parameters including vision_message

        Returns:
            Chat response with accumulated content and usage
        """
        start_time = time.time()
        handler = AsyncStreamHandler(on_chunk)

        try:
            generation_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
            }

            if kwargs.get("max_tokens") or self.config.max_tokens:
                generation_config["max_output_tokens"] = (
                    kwargs.get("max_tokens") or self.config.max_tokens
                )

            # Build message content
            vision_message = kwargs.get("vision_message")
            if vision_message and isinstance(vision_message, VisionMessage):
                content = self._build_vision_content(vision_message)
            else:
                content = message

            response = await self.model.generate_content_async(
                content,
                generation_config=generation_config,
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    await handler.handle_chunk(chunk.text)

            latency_ms = (time.time() - start_time) * 1000

            prompt_tokens = (
                response.usage_metadata.prompt_character_count if response.usage_metadata else 0
            )
            completion_tokens = (
                response.usage_metadata.candidates_token_count if response.usage_metadata else 0
            )

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
                finish_reason=(
                    response.candidates[0].finish_reason.name if response.candidates else None
                ),
            )

        except Exception as e:
            self._handle_google_error(e, "astream")

    def _handle_google_error(self, error: Exception, operation: str) -> None:
        """
        Convert Google SDK errors to Nexus exceptions.

        Args:
            error: Google SDK error
            operation: Operation that failed

        Raises:
            Appropriate NexusError subclass
        """
        error_msg = str(error)

        # Map Google errors
        if (
            "rate_limit" in error_msg.lower()
            or "429" in error_msg
            or "RESOURCE_EXHAUSTED" in error_msg
        ):
            raise RateLimitError(f"Google rate limit exceeded: {error_msg}")
        elif (
            "invalid_api_key" in error_msg or "403" in error_msg or "PERMISSION_DENIED" in error_msg
        ):
            raise AuthenticationError(f"Invalid Google API key: {error_msg}")
        elif (
            "invalid_request" in error_msg or "400" in error_msg or "INVALID_ARGUMENT" in error_msg
        ):
            raise InvalidRequestError(f"Invalid request to Google: {error_msg}")
        else:
            raise ProviderError(
                f"{operation} failed with Google: {error_msg}",
                error_code="GOOGLE_ERROR",
            )
