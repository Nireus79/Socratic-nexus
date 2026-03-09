"""Base provider interface for Socrates Nexus."""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from ..models import ChatResponse, TokenUsage, LLMConfig


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        """Initialize provider with config."""
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def chat(self, message: str, **kwargs) -> ChatResponse:
        """
        Send a chat message and get response.

        Args:
            message: User message
            **kwargs: Additional provider-specific arguments

        Returns:
            ChatResponse with content and usage information
        """
        pass

    @abstractmethod
    async def achat(self, message: str, **kwargs) -> ChatResponse:
        """Async version of chat."""
        pass

    @abstractmethod
    def stream(
        self, message: str, on_chunk: Callable[[str], None], **kwargs
    ) -> ChatResponse:
        """
        Stream a chat response with callback.

        Args:
            message: User message
            on_chunk: Callback function for each chunk
            **kwargs: Additional provider-specific arguments

        Returns:
            ChatResponse with complete content and usage
        """
        pass

    @abstractmethod
    async def astream(
        self, message: str, on_chunk: Callable[[str], None], **kwargs
    ) -> ChatResponse:
        """Async version of stream."""
        pass

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate that provider credentials are correct."""
        pass

    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate cost for the usage. Override in subclasses."""
        return 0.0

    def __repr__(self) -> str:
        return f"{self.name}(model={self.config.model})"
