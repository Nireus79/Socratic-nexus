"""
LangChain integration for Socratic Nexus clients.

Provides a wrapper that makes Socratic Nexus clients compatible with LangChain's LLM interface.

Example:
    from socratic_nexus.clients import ClaudeClient
    from socratic_nexus.integrations.langchain import SocratesNexusLLM
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    client = ClaudeClient(api_key="your-api-key")
    llm = SocratesNexusLLM(client=client)

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a summary about {topic}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(topic="Machine Learning")
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.clients.google_client import GoogleClient
from socratic_nexus.clients.openai_client import OpenAIClient
from socratic_nexus.clients.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class SocratesNexusLLM:
    """
    LangChain-compatible wrapper for Socratic Nexus clients.

    This class implements the interface needed to use Socratic Nexus clients
    within LangChain chains and agents.

    Attributes:
        client: The underlying Socratic Nexus client (Claude, OpenAI, Google, or Ollama)
        model: The model name being used
        temperature: Temperature parameter for generation (0-2)
        max_tokens: Maximum tokens in the response
    """

    def __init__(
        self,
        client: ClaudeClient | OpenAIClient | GoogleClient | OllamaClient,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize SocratesNexusLLM wrapper.

        Args:
            client: The Socratic Nexus client to wrap (required)
            temperature: Temperature parameter for generation (default: 0.7)
            max_tokens: Maximum tokens in response (optional)
            **kwargs: Additional arguments (ignored for compatibility)

        Raises:
            ValueError: If client is None or not a recognized Socratic Nexus client
        """
        if client is None:
            raise ValueError("client parameter is required")

        if not isinstance(
            client,
            (ClaudeClient, OpenAIClient, GoogleClient, OllamaClient),
        ):
            raise ValueError(
                f"client must be a Socratic Nexus client, got {type(client).__name__}"
            )

        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = getattr(client, "model", "unknown")
        self._call_count = 0

    def __call__(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Call the LLM with the given prompt.

        Args:
            prompt: The input prompt
            stop: Stop sequences (not used by all clients)
            **kwargs: Additional arguments passed to the client

        Returns:
            The generated response text
        """
        return self._generate(prompt, stop=stop, **kwargs)

    def _generate(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Generate a response using the underlying Socratic Nexus client.

        Args:
            prompt: The input prompt
            stop: Stop sequences (ignored for compatibility)
            **kwargs: Additional arguments

        Returns:
            The generated response text
        """
        try:
            self._call_count += 1
            logger.debug(f"SocratesNexusLLM call #{self._call_count}: {prompt[:100]}...")

            # Use the client's generate_response method
            response = self.client.generate_response(prompt)

            # Handle both string and ChatResponse object responses
            if isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                # ChatResponse object
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error in SocratesNexusLLM: {e}")
            raise

    async def _generate_async(
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        """
        Generate a response asynchronously.

        Args:
            prompt: The input prompt
            stop: Stop sequences (ignored for compatibility)
            **kwargs: Additional arguments

        Returns:
            The generated response text
        """
        try:
            logger.debug(f"SocratesNexusLLM async call: {prompt[:100]}...")

            # Check if client has async method
            if hasattr(self.client, "generate_response_async"):
                response = await self.client.generate_response_async(prompt)
            else:
                # Fallback to sync method
                response = self.client.generate_response(prompt)

            # Handle both string and ChatResponse object responses
            if isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error in SocratesNexusLLM async: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM for LangChain identification."""
        return f"socratic_nexus_{type(self.client).__name__.lower()}"

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters for debugging and logging."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "client_type": type(self.client).__name__,
        }

    def get_num_tokens(self, text: str) -> int:
        """
        Get the number of tokens in the given text.

        Uses a simple heuristic: ~4 characters = 1 token.

        Args:
            text: The text to tokenize

        Returns:
            Approximate number of tokens
        """
        return len(text) // 4

    def get_token_ids(self, text: str) -> List[int]:
        """
        Get token IDs for the given text.

        Returns placeholder token IDs.

        Args:
            text: The text to tokenize

        Returns:
            List of placeholder token IDs
        """
        return list(range(self.get_num_tokens(text)))


# For LangChain 0.1.0+ compatibility
try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun

    class SocratesNexusLLMOfficial(LLM):
        """
        Official LangChain-compatible LLM wrapper for Socratic Nexus.

        This version implements LangChain's official LLM interface for
        seamless integration with LangChain chains and agents.
        """

        client: Any  # The Socratic Nexus client
        model: str = "unknown"
        temperature: float = 0.7
        max_tokens: Optional[int] = None

        def __init__(self, client: Any, **kwargs: Any):
            """Initialize with a Socratic Nexus client."""
            super().__init__(**kwargs)
            self.client = client
            self.model = getattr(client, "model", "unknown")

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            """Call the LLM."""
            response = self.client.generate_response(prompt)
            if isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                return response.content
            else:
                return str(response)

        @property
        def _llm_type(self) -> str:
            """Return the type of LLM."""
            return f"socratic_nexus_{type(self.client).__name__.lower()}"

        @property
        def _identifying_params(self) -> dict:
            """Return identifying parameters."""
            return {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

except ImportError:
    # LangChain not installed or version doesn't have this interface
    SocratesNexusLLMOfficial = None  # type: ignore[misc,assignment]


__all__ = ["SocratesNexusLLM"]
