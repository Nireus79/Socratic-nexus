"""
Openclaw integration for Socratic Nexus clients.

Provides skill classes for using Socratic Nexus clients as Openclaw skills.

Example:
    from socratic_nexus.clients import ClaudeClient
    from socratic_nexus.integrations.openclaw import NexusLLMSkill

    client = ClaudeClient(api_key="your-api-key")
    skill = NexusLLMSkill(
        client=client,
        name="analyze",
        description="Analyze text using Claude"
    )

    result = skill.query("Analyze this text...")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from socratic_nexus.clients.claude_client import ClaudeClient
from socratic_nexus.clients.google_client import GoogleClient
from socratic_nexus.clients.openai_client import OpenAIClient
from socratic_nexus.clients.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class NexusLLMSkill:
    """
    Openclaw skill for using Socratic Nexus clients.

    This class provides a skill interface that can be used with Openclaw
    to integrate Socratic Nexus LLM clients into Openclaw workflows.

    Attributes:
        name: The skill name
        description: The skill description
        client: The underlying Socratic Nexus client
        model: The model being used
    """

    def __init__(
        self,
        client: ClaudeClient | OpenAIClient | GoogleClient | OllamaClient,
        name: str = "nexus_llm_skill",
        description: str = "Use Socratic Nexus LLM client",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize NexusLLMSkill.

        Args:
            client: The Socratic Nexus client to wrap (required)
            name: The skill name
            description: The skill description
            system_prompt: Optional system prompt to prepend to all queries
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens in response
            metadata: Optional metadata dict

        Raises:
            ValueError: If client is None or not a recognized Socratic Nexus client
        """
        if client is None:
            raise ValueError("client parameter is required")

        if not isinstance(
            client,
            (ClaudeClient, OpenAIClient, GoogleClient, OllamaClient),
        ):
            raise ValueError(f"client must be a Socratic Nexus client, got {type(client).__name__}")

        self.client = client
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.metadata = metadata or {}
        self.model = getattr(client, "model", "unknown")
        self._call_count = 0

    def query(self, prompt: str, **kwargs: Any) -> str:
        """
        Execute the skill with the given prompt.

        Args:
            prompt: The input prompt/query
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            The generated response
        """
        try:
            self._call_count += 1
            logger.debug(f"NexusLLMSkill.query #{self._call_count}: {prompt[:100]}...")

            # Prepare full prompt with system message if provided
            if self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Call the client
            response = self.client.generate_response(full_prompt)

            # Extract text from response
            if isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error in NexusLLMSkill.query: {e}")
            raise

    async def query_async(self, prompt: str, **kwargs: Any) -> str:
        """
        Execute the skill asynchronously.

        Args:
            prompt: The input prompt/query
            **kwargs: Additional arguments

        Returns:
            The generated response
        """
        try:
            logger.debug(f"NexusLLMSkill.query_async: {prompt[:100]}...")

            # Prepare full prompt
            if self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Check if client has async method
            if hasattr(self.client, "generate_response_async"):
                response = await self.client.generate_response_async(full_prompt)
            else:
                # Fallback to sync method
                response = self.client.generate_response(full_prompt)

            # Extract text from response
            if isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error in NexusLLMSkill.query_async: {e}")
            raise

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process structured input and return structured output.

        Implements Openclaw's process interface for compatibility with
        more complex skill workflows.

        Args:
            input_data: Dictionary containing:
                - 'prompt' or 'input': The prompt to process
                - 'context' (optional): Additional context
                - 'instructions' (optional): Special instructions

        Returns:
            Dictionary with:
                - 'output': The generated response
                - 'success': Whether the operation succeeded
                - 'metadata': Any metadata about the operation
        """
        try:
            # Extract prompt
            prompt = input_data.get("prompt") or input_data.get("input", "")

            if not prompt:
                return {
                    "output": "",
                    "success": False,
                    "error": "No prompt provided",
                    "metadata": {},
                }

            # Add context if provided
            if "context" in input_data:
                prompt = f"Context: {input_data['context']}\n\nQuery: {prompt}"

            # Add instructions if provided
            if "instructions" in input_data:
                prompt = f"{input_data['instructions']}\n\n{prompt}"

            # Generate response
            response = self.query(prompt)

            return {
                "output": response,
                "success": True,
                "metadata": {
                    "model": self.model,
                    "call_count": self._call_count,
                },
            }

        except Exception as e:
            logger.error(f"Error in NexusLLMSkill.process: {e}")
            return {
                "output": "",
                "success": False,
                "error": str(e),
                "metadata": {},
            }

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this skill for Openclaw registration.

        Returns:
            Dictionary with skill metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": "llm",
            "client_type": type(self.client).__name__,
            "model": self.model,
            "capabilities": [
                "text_generation",
                "analysis",
                "code_generation",
                "question_answering",
            ],
            "parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """Return string representation of the skill."""
        return f"NexusLLMSkill(name={self.name}, model={self.model}, client={type(self.client).__name__})"

    def __repr__(self) -> str:
        """Return repr of the skill."""
        return self.__str__()


class NexusAnalysisSkill(NexusLLMSkill):
    """
    Specialized Openclaw skill for text analysis using Socratic Nexus.

    This skill is optimized for analyzing text, extracting insights,
    and generating summaries.
    """

    def __init__(self, client: Any, **kwargs: Any):
        """Initialize the analysis skill."""
        kwargs.setdefault("name", "nexus_analysis_skill")
        kwargs.setdefault("description", "Analyze and extract insights from text")
        kwargs.setdefault(
            "system_prompt",
            "You are an expert analyst. Analyze the provided text and extract key insights, patterns, and recommendations.",
        )
        super().__init__(client=client, **kwargs)

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze the given text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with analysis results
        """
        prompt = f"Analyze this text:\n\n{text}"
        response = self.query(prompt)

        return {
            "original_text": text,
            "analysis": response,
            "model": self.model,
        }


class NexusCodeGenSkill(NexusLLMSkill):
    """
    Specialized Openclaw skill for code generation using Socratic Nexus.

    This skill is optimized for generating code, implementing features,
    and debugging issues.
    """

    def __init__(self, client: Any, **kwargs: Any):
        """Initialize the code generation skill."""
        kwargs.setdefault("name", "nexus_codegen_skill")
        kwargs.setdefault("description", "Generate and review code")
        kwargs.setdefault(
            "system_prompt",
            "You are an expert programmer. Generate clean, well-documented code that follows best practices.",
        )
        super().__init__(client=client, **kwargs)

    def generate_code(self, specification: str, language: str = "python") -> Dict[str, Any]:
        """
        Generate code based on specification.

        Args:
            specification: What code to generate
            language: Programming language (default: python)

        Returns:
            Dictionary with generated code
        """
        prompt = f"Generate {language} code for: {specification}"
        response = self.query(prompt)

        return {
            "specification": specification,
            "language": language,
            "code": response,
            "model": self.model,
        }


class NexusDocumentationSkill(NexusLLMSkill):
    """
    Specialized Openclaw skill for documentation generation.

    This skill is optimized for creating documentation, writing guides,
    and generating technical explanations.
    """

    def __init__(self, client: Any, **kwargs: Any):
        """Initialize the documentation skill."""
        kwargs.setdefault("name", "nexus_docs_skill")
        kwargs.setdefault("description", "Generate documentation and guides")
        kwargs.setdefault(
            "system_prompt",
            "You are a technical writer. Create clear, comprehensive documentation with examples.",
        )
        super().__init__(client=client, **kwargs)

    def generate_documentation(self, subject: str, doc_type: str = "guide") -> Dict[str, Any]:
        """
        Generate documentation for the given subject.

        Args:
            subject: What to document
            doc_type: Type of documentation (guide, api, tutorial, etc.)

        Returns:
            Dictionary with generated documentation
        """
        prompt = f"Write a {doc_type} for: {subject}"
        response = self.query(prompt)

        return {
            "subject": subject,
            "type": doc_type,
            "documentation": response,
            "model": self.model,
        }


__all__ = [
    "NexusLLMSkill",
    "NexusAnalysisSkill",
    "NexusCodeGenSkill",
    "NexusDocumentationSkill",
]
