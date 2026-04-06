"""Vision model support for multimodal image-text understanding."""

from dataclasses import dataclass
from typing import List, Optional, Union
from .models import TextContent, ImageContent
from .utils.images import (
    encode_image_base64,
    detect_media_type,
    validate_image_format,
    is_image_url,
    is_image_path,
)


@dataclass
class VisionMessage:
    """Represents a message that may contain text and images."""

    text: str
    images: Optional[List[Union[str, bytes]]] = None

    def __post_init__(self):
        """Validate vision message."""
        if self.images is None:
            self.images = []


class VisionProcessor:
    """Processes images for vision models."""

    @staticmethod
    def prepare_image(source: Union[str, bytes], detail: Optional[str] = None) -> ImageContent:
        """
        Prepare image for sending to vision model.

        Args:
            source: Image URL, file path, or bytes
            detail: Detail level ("low", "high") for supported models

        Returns:
            ImageContent object

        Raises:
            ValueError: If source is invalid
            FileNotFoundError: If file path doesn't exist
        """
        if isinstance(source, bytes):
            # Direct bytes - encode to base64
            return ImageContent(
                type="image",
                source=f"data:application/octet-stream;base64,{source.hex()}",
                media_type="image/jpeg",
                detail=detail,
            )

        if is_image_url(source):
            # URL - use directly
            return ImageContent(type="image", source=source, media_type="image/jpeg", detail=detail)

        if is_image_path(source):
            # File path - validate and encode
            if not validate_image_format(source):
                raise ValueError(f"Unsupported image format: {source}")

            media_type = detect_media_type(source)
            encoded = encode_image_base64(source)

            return ImageContent(
                type="image",
                source=f"data:{media_type};base64,{encoded}",
                media_type=media_type,
                detail=detail,
            )

        raise ValueError(f"Invalid image source: {source}")

    @staticmethod
    def prepare_multimodal_message(
        vision_message: VisionMessage, detail: Optional[str] = None
    ) -> List[Union[TextContent, ImageContent]]:
        """
        Convert VisionMessage to content blocks.

        Args:
            vision_message: VisionMessage with text and images
            detail: Detail level for images

        Returns:
            List of TextContent and ImageContent blocks

        Raises:
            ValueError: If image preparation fails
        """
        contents = []

        # Add text first
        if vision_message.text:
            contents.append(TextContent(type="text", text=vision_message.text))

        # Add images
        if vision_message.images:
            for image_source in vision_message.images:
                image_content = VisionProcessor.prepare_image(image_source, detail)
                contents.append(image_content)

        return contents


class VisionCapabilities:
    """Manages vision capabilities across providers."""

    # Vision model identifiers
    VISION_MODELS = {
        "openai": [
            "gpt-4-vision-preview",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "google": [
            "gemini-1.5-pro-vision",
            "gemini-1.5-flash-vision",
            "gemini-pro-vision",
        ],
    }

    # Supported image formats per provider
    SUPPORTED_FORMATS = {
        "openai": ["jpeg", "png", "gif", "webp"],
        "google": ["jpeg", "png", "gif", "webp"],
    }

    # Maximum image dimensions
    MAX_DIMENSIONS = {
        "openai": {"width": 2000, "height": 2000},
        "google": {"width": 3840, "height": 2160},
    }

    @staticmethod
    def is_vision_model(provider: str, model: str) -> bool:
        """
        Check if model supports vision.

        Args:
            provider: Provider name
            model: Model identifier

        Returns:
            True if model supports vision, False otherwise
        """
        models = VisionCapabilities.VISION_MODELS.get(provider, [])
        return model in models

    @staticmethod
    def get_vision_models(provider: str) -> List[str]:
        """
        Get list of vision models for provider.

        Args:
            provider: Provider name

        Returns:
            List of vision model identifiers
        """
        return VisionCapabilities.VISION_MODELS.get(provider, [])

    @staticmethod
    def validate_image_format(provider: str, format_name: str) -> bool:
        """
        Validate image format for provider.

        Args:
            provider: Provider name
            format_name: Image format (without dot)

        Returns:
            True if format is supported, False otherwise
        """
        supported = VisionCapabilities.SUPPORTED_FORMATS.get(provider, [])
        return format_name.lower() in supported
