"""Tests for vision model support."""

import pytest
from socrates_nexus.vision import (
    VisionMessage,
    VisionProcessor,
    VisionCapabilities,
)
from socrates_nexus.models import ImageContent, TextContent


class TestVisionMessage:
    """Test VisionMessage dataclass."""

    def test_vision_message_creation(self):
        """Test creating a vision message."""
        msg = VisionMessage(text="Describe this image")
        assert msg.text == "Describe this image"
        assert msg.images == []

    def test_vision_message_with_images(self):
        """Test vision message with images."""
        msg = VisionMessage(
            text="What's in this image?",
            images=["image1.jpg", "image2.jpg"],
        )
        assert msg.text == "What's in this image?"
        assert len(msg.images) == 2
        assert "image1.jpg" in msg.images

    def test_vision_message_post_init(self):
        """Test that post_init initializes empty images list."""
        msg = VisionMessage(text="Test")
        assert msg.images is not None
        assert isinstance(msg.images, list)


class TestVisionProcessor:
    """Test VisionProcessor class."""

    def test_prepare_image_with_bytes(self):
        """Test preparing image from bytes."""
        image_bytes = b"fake_image_data"
        result = VisionProcessor.prepare_image(image_bytes)

        assert isinstance(result, ImageContent)
        assert result.type == "image"
        assert "data:" in result.source
        assert "base64" in result.source

    def test_prepare_image_with_url(self):
        """Test preparing image from URL."""
        url = "https://example.com/image.jpg"
        result = VisionProcessor.prepare_image(url)

        assert isinstance(result, ImageContent)
        assert result.type == "image"
        assert result.source == url
        assert result.media_type == "image/jpeg"

    def test_prepare_image_with_detail_level(self):
        """Test preparing image with detail level."""
        url = "https://example.com/image.jpg"
        result = VisionProcessor.prepare_image(url, detail="high")

        assert result.detail == "high"

    def test_prepare_image_invalid_source(self):
        """Test preparing image with invalid source."""
        with pytest.raises(ValueError):
            VisionProcessor.prepare_image("invalid_source")

    def test_prepare_multimodal_message_text_only(self):
        """Test preparing multimodal message with only text."""
        msg = VisionMessage(text="Describe this")
        result = VisionProcessor.prepare_multimodal_message(msg)

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Describe this"

    def test_prepare_multimodal_message_with_image_urls(self):
        """Test preparing multimodal message with image URLs."""
        msg = VisionMessage(
            text="Describe these images",
            images=[
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
            ],
        )
        result = VisionProcessor.prepare_multimodal_message(msg)

        assert len(result) == 3  # 1 text + 2 images
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Describe these images"
        assert isinstance(result[1], ImageContent)
        assert isinstance(result[2], ImageContent)

    def test_prepare_multimodal_message_empty_text(self):
        """Test preparing multimodal message with empty text."""
        msg = VisionMessage(
            text="",
            images=["https://example.com/image.jpg"],
        )
        result = VisionProcessor.prepare_multimodal_message(msg)

        # Should not include text if empty
        assert all(isinstance(item, ImageContent) for item in result)

    def test_prepare_multimodal_message_with_detail(self):
        """Test preparing multimodal message with detail level."""
        msg = VisionMessage(
            text="Analyze this image",
            images=["https://example.com/image.jpg"],
        )
        result = VisionProcessor.prepare_multimodal_message(msg, detail="low")

        assert len(result) == 2
        image_content = result[1]
        assert image_content.detail == "low"


class TestVisionCapabilities:
    """Test VisionCapabilities class."""

    def test_vision_models_structure(self):
        """Test vision models dictionary structure."""
        models = VisionCapabilities.VISION_MODELS
        assert "openai" in models
        assert "google" in models
        assert len(models["openai"]) > 0
        assert len(models["google"]) > 0

    def test_is_vision_model_openai(self):
        """Test checking vision capability for OpenAI models."""
        assert VisionCapabilities.is_vision_model("openai", "gpt-4-vision-preview")
        assert VisionCapabilities.is_vision_model("openai", "gpt-4o")
        assert not VisionCapabilities.is_vision_model("openai", "gpt-3.5-turbo")

    def test_is_vision_model_google(self):
        """Test checking vision capability for Google models."""
        assert VisionCapabilities.is_vision_model("google", "gemini-pro-vision")
        assert VisionCapabilities.is_vision_model("google", "gemini-1.5-pro-vision")
        assert not VisionCapabilities.is_vision_model("google", "gemini-pro")

    def test_is_vision_model_unknown_provider(self):
        """Test checking vision for unknown provider."""
        result = VisionCapabilities.is_vision_model("unknown", "some-model")
        assert result is False

    def test_get_vision_models_openai(self):
        """Test getting OpenAI vision models."""
        models = VisionCapabilities.get_vision_models("openai")
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4o" in models

    def test_get_vision_models_google(self):
        """Test getting Google vision models."""
        models = VisionCapabilities.get_vision_models("google")
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gemini-pro-vision" in models

    def test_get_vision_models_unknown_provider(self):
        """Test getting models for unknown provider."""
        models = VisionCapabilities.get_vision_models("unknown")
        assert models == []

    def test_supported_formats_structure(self):
        """Test supported image formats structure."""
        formats = VisionCapabilities.SUPPORTED_FORMATS
        assert "openai" in formats
        assert "google" in formats
        assert "jpeg" in formats["openai"]
        assert "png" in formats["google"]

    def test_validate_image_format_openai(self):
        """Test image format validation for OpenAI."""
        assert VisionCapabilities.validate_image_format("openai", "jpeg")
        assert VisionCapabilities.validate_image_format("openai", "png")
        assert VisionCapabilities.validate_image_format("openai", "gif")
        assert VisionCapabilities.validate_image_format("openai", "webp")
        assert not VisionCapabilities.validate_image_format("openai", "bmp")
        assert not VisionCapabilities.validate_image_format("openai", "tiff")

    def test_validate_image_format_google(self):
        """Test image format validation for Google."""
        assert VisionCapabilities.validate_image_format("google", "jpeg")
        assert VisionCapabilities.validate_image_format("google", "png")
        assert not VisionCapabilities.validate_image_format("google", "bmp")

    def test_validate_image_format_case_insensitive(self):
        """Test that format validation is case-insensitive."""
        assert VisionCapabilities.validate_image_format("openai", "JPEG")
        assert VisionCapabilities.validate_image_format("openai", "PNG")
        assert VisionCapabilities.validate_image_format("openai", "JpEg")

    def test_validate_image_format_unknown_provider(self):
        """Test format validation for unknown provider."""
        result = VisionCapabilities.validate_image_format("unknown", "jpeg")
        assert result is False

    def test_max_dimensions_structure(self):
        """Test max dimensions structure."""
        dims = VisionCapabilities.MAX_DIMENSIONS
        assert "openai" in dims
        assert "google" in dims
        assert dims["openai"]["width"] == 2000
        assert dims["openai"]["height"] == 2000
        assert dims["google"]["width"] == 3840
        assert dims["google"]["height"] == 2160

    def test_vision_models_content(self):
        """Test specific vision models are present."""
        openai_models = VisionCapabilities.get_vision_models("openai")
        google_models = VisionCapabilities.get_vision_models("google")

        # Check for specific known models
        assert any("gpt-4" in m for m in openai_models)
        assert any("gemini" in m for m in google_models)

    def test_vision_processor_prepare_image_with_bytes_and_detail(self):
        """Test preparing image from bytes with detail level."""
        image_bytes = b"test_data"
        result = VisionProcessor.prepare_image(image_bytes, detail="high")

        assert result.detail == "high"
        assert "base64" in result.source

    def test_multimodal_message_with_url_and_bytes(self):
        """Test preparing multimodal message with mixed sources."""
        # This tests the integration of both URL and bytes handling
        image_bytes = b"test_image_data"
        url = "https://example.com/image.jpg"

        msg = VisionMessage(
            text="Check both images",
            images=[url, image_bytes],
        )

        result = VisionProcessor.prepare_multimodal_message(msg)

        # Should have text + 2 images
        assert len(result) == 3
        assert isinstance(result[0], TextContent)
        assert isinstance(result[1], ImageContent)
        assert isinstance(result[2], ImageContent)

    def test_vision_message_none_images_converted_to_list(self):
        """Test that None images are converted to empty list."""
        msg = VisionMessage(text="Test", images=None)
        assert msg.images == []
