"""Tests for vision models and function calling."""

from unittest.mock import patch

from socratic_nexus.models import (
    ChatResponse,
    TokenUsage,
    ImageContent,
    TextContent,
    Tool,
    Function,
    FunctionCall,
    ToolCall,
)
from socratic_nexus.client import LLMClient
from socratic_nexus.utils.images import (
    detect_media_type,
    validate_image_format,
    is_image_url,
)


class TestImageUtilities:
    """Test image utility functions."""

    def test_detect_media_type_jpeg(self):
        """Test JPEG media type detection."""
        assert detect_media_type("image.jpg") == "image/jpeg"
        assert detect_media_type("image.jpeg") == "image/jpeg"

    def test_detect_media_type_png(self):
        """Test PNG media type detection."""
        assert detect_media_type("image.png") == "image/png"

    def test_detect_media_type_gif(self):
        """Test GIF media type detection."""
        assert detect_media_type("image.gif") == "image/gif"

    def test_detect_media_type_webp(self):
        """Test WebP media type detection."""
        assert detect_media_type("image.webp") == "image/webp"

    def test_detect_media_type_case_insensitive(self):
        """Test media type detection is case insensitive."""
        assert detect_media_type("IMAGE.JPG") == "image/jpeg"
        assert detect_media_type("Image.PNG") == "image/png"

    def test_validate_image_format_supported(self):
        """Test validation of supported formats."""
        assert validate_image_format("image.jpg") is True
        assert validate_image_format("image.png") is True
        assert validate_image_format("image.gif") is True
        assert validate_image_format("image.webp") is True

    def test_validate_image_format_unsupported(self):
        """Test validation of unsupported formats."""
        assert validate_image_format("image.txt") is False
        assert validate_image_format("image.pdf") is False
        assert validate_image_format("image.doc") is False

    def test_is_image_url_http(self):
        """Test URL detection for http."""
        assert is_image_url("http://example.com/image.jpg") is True

    def test_is_image_url_https(self):
        """Test URL detection for https."""
        assert is_image_url("https://example.com/image.jpg") is True

    def test_is_image_url_file_path(self):
        """Test URL detection for file paths."""
        assert is_image_url("image.jpg") is False
        assert is_image_url("/path/to/image.jpg") is False


class TestImageContent:
    """Test ImageContent model."""

    def test_image_content_creation(self):
        """Test creating ImageContent."""
        content = ImageContent(source="https://example.com/image.jpg", media_type="image/jpeg")
        assert content.type == "image"
        assert content.source == "https://example.com/image.jpg"
        assert content.media_type == "image/jpeg"

    def test_image_content_with_file_path(self):
        """Test ImageContent with file path."""
        content = ImageContent(source="/path/to/image.png", media_type="image/png")
        assert content.source == "/path/to/image.png"

    def test_image_content_with_base64(self):
        """Test ImageContent with base64 data."""
        base64_data = "iVBORw0KGgoAAAANSUhEUg="
        content = ImageContent(source=base64_data, media_type="image/png")
        assert content.source == base64_data

    def test_image_content_with_detail(self):
        """Test ImageContent with detail level."""
        content = ImageContent(source="https://example.com/image.jpg", detail="high")
        assert content.detail == "high"


class TestTextContent:
    """Test TextContent model."""

    def test_text_content_creation(self):
        """Test creating TextContent."""
        content = TextContent(text="Hello, world!")
        assert content.type == "text"
        assert content.text == "Hello, world!"

    def test_text_content_empty(self):
        """Test TextContent with empty text."""
        content = TextContent(text="")
        assert content.text == ""


class TestVisionSupport:
    """Test vision model support."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_chat_with_image_url(self, mock_chat):
        """Test chat with image URL."""
        mock_chat.return_value = ChatResponse(
            content="The image shows a cat",
            provider="anthropic",
            model="claude-3-5-sonnet",
            usage=TokenUsage(100, 20, 120, 0.001, "anthropic", "claude-3-5-sonnet"),
        )

        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet", api_key="test")

        # Simulate chat with image
        response = client.chat("What's in this image?", image_url="https://example.com/image.jpg")

        assert response.content == "The image shows a cat"

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_chat_with_multiple_images(self, mock_chat):
        """Test chat with multiple images."""
        mock_chat.return_value = ChatResponse(
            content="Comparison complete",
            provider="anthropic",
            model="claude-3-5-sonnet",
            usage=TokenUsage(150, 10, 160, 0.002, "anthropic", "claude-3-5-sonnet"),
        )

        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet", api_key="test")

        response = client.chat(
            "Compare these images",
            images=["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
        )

        assert response.content == "Comparison complete"

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_chat_with_mixed_content(self, mock_chat):
        """Test chat with mixed text and image content."""
        mock_chat.return_value = ChatResponse(
            content="Analysis complete",
            provider="anthropic",
            model="claude-3-5-sonnet",
            usage=TokenUsage(200, 30, 230, 0.002, "anthropic", "claude-3-5-sonnet"),
        )

        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet", api_key="test")

        response = client.chat(
            "Analyze this image and tell me what you see", image="https://example.com/image.jpg"
        )

        assert response.content == "Analysis complete"


class TestFunctionCalling:
    """Test function calling / tool use support."""

    def test_function_definition_creation(self):
        """Test creating function definition."""
        func = Function(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        )
        assert func.name == "get_weather"
        assert "location" in func.parameters["properties"]

    def test_tool_definition_creation(self):
        """Test creating tool definition."""
        tool = Tool(function=Function(name="get_weather", description="Get weather", parameters={}))
        assert tool.type == "function"
        assert tool.function.name == "get_weather"

    def test_function_call_creation(self):
        """Test creating function call."""
        call = FunctionCall(name="get_weather", arguments='{"location": "New York"}')
        assert call.name == "get_weather"
        assert "New York" in call.arguments

    def test_tool_call_creation(self):
        """Test creating tool call."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="get_weather", arguments='{"location": "Paris"}'),
        )
        assert tool_call.id == "call_123"
        assert tool_call.function.name == "get_weather"

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_chat_with_tools(self, mock_chat):
        """Test chat with function calling."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="get_weather", arguments='{"location": "New York"}'),
        )

        mock_chat.return_value = ChatResponse(
            content="I'll check the weather for you",
            provider="anthropic",
            model="claude-3-5-sonnet",
            usage=TokenUsage(50, 25, 75, 0.001, "anthropic", "claude-3-5-sonnet"),
            tool_calls=[tool_call],
        )

        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet", api_key="test")

        tools = [
            Tool(
                function=Function(
                    name="get_weather",
                    description="Get weather for a location",
                    parameters={"type": "object", "properties": {"location": {"type": "string"}}},
                )
            )
        ]

        response = client.chat("What's the weather in New York?", tools=tools)

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].function.name == "get_weather"

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_chat_with_multiple_tools(self, mock_chat):
        """Test chat with multiple tools available."""
        mock_chat.return_value = ChatResponse(
            content="I'll help with that",
            provider="anthropic",
            model="claude-3-5-sonnet",
            usage=TokenUsage(100, 50, 150, 0.002, "anthropic", "claude-3-5-sonnet"),
            tool_calls=[],
        )

        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet", api_key="test")

        tools = [
            Tool(
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {}},
                )
            ),
            Tool(
                function=Function(
                    name="get_time",
                    description="Get current time",
                    parameters={"type": "object", "properties": {}},
                )
            ),
        ]

        response = client.chat("Get me weather and time", tools=tools)

        assert response.tool_calls is not None

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_tool_choice_auto(self, mock_chat):
        """Test tool_choice=auto mode."""
        mock_chat.return_value = ChatResponse(
            content="Response",
            provider="anthropic",
            model="claude-3-5-sonnet",
            usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3-5-sonnet"),
            tool_calls=[],
        )

        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet", api_key="test")

        tools = [
            Tool(function=Function(name="test", description="Test", parameters={"type": "object"}))
        ]

        response = client.chat("Test", tools=tools, tool_choice="auto")

        assert response is not None

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_tool_choice_none(self, mock_chat):
        """Test tool_choice=none mode (no tool use)."""
        mock_chat.return_value = ChatResponse(
            content="I cannot use tools",
            provider="anthropic",
            model="claude-3-5-sonnet",
            usage=TokenUsage(0, 0, 0, 0.0, "anthropic", "claude-3-5-sonnet"),
            tool_calls=None,
        )

        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet", api_key="test")

        tools = [
            Tool(function=Function(name="test", description="Test", parameters={"type": "object"}))
        ]

        response = client.chat("Test", tools=tools, tool_choice="none")

        assert response.tool_calls is None


class TestVisionAndFunctionsCombined:
    """Test vision and function calling together."""

    @patch("socratic_nexus.providers.anthropic.AnthropicProvider.chat")
    def test_vision_with_tool_use(self, mock_chat):
        """Test combining vision and function calling."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="analyze_image", arguments='{"aspect": "objects"}'),
        )

        mock_chat.return_value = ChatResponse(
            content="Analyzing image",
            provider="anthropic",
            model="claude-3-5-sonnet",
            usage=TokenUsage(150, 50, 200, 0.002, "anthropic", "claude-3-5-sonnet"),
            tool_calls=[tool_call],
        )

        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet", api_key="test")

        tools = [
            Tool(
                function=Function(
                    name="analyze_image", description="Analyze image", parameters={"type": "object"}
                )
            )
        ]

        response = client.chat(
            "Analyze this image for me", image="https://example.com/image.jpg", tools=tools
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
