"""
LangChain Integration for Socrates Nexus.

Provides SocratesNexusLLM as a LangChain-compatible LLM provider
with multi-provider support and automatic retry logic.
"""

from typing import Any, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from socrates_nexus import LLMClient, LLMConfig


class SocratesNexusLLM(LLM):
    """
    LangChain LLM wrapper for Socrates Nexus.

    Provides a drop-in replacement for OpenAI LLM with multi-provider support,
    automatic retry logic, and token tracking.

    Supported providers: anthropic, openai, google, ollama

    Example:
        >>> from socrates_nexus.integrations.langchain import SocratesNexusLLM
        >>> from langchain.chains import LLMChain
        >>> from langchain.prompts import PromptTemplate
        >>>
        >>> llm = SocratesNexusLLM(provider="anthropic", model="claude-opus")
        >>> prompt = PromptTemplate(template="What is {topic}?", input_variables=["topic"])
        >>> chain = LLMChain(llm=llm, prompt=prompt)
        >>> result = chain.run(topic="machine learning")
        >>>
        >>> # Easy provider switching
        >>> llm2 = SocratesNexusLLM(provider="openai", model="gpt-4")
        >>> chain2 = LLMChain(llm=llm2, prompt=prompt)
    """

    provider: str = "anthropic"
    model: str = "claude-opus"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    retry_attempts: int = 3
    cache_responses: bool = True
    cache_ttl: int = 300

    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return "socrates_nexus"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the LLM.

        Args:
            prompt: The prompt to send to the LLM
            stop: Stop sequences (not used by all providers)
            run_manager: LangChain callback manager
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        config = LLMConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            retry_attempts=self.retry_attempts,
            cache_responses=self.cache_responses,
            cache_ttl=self.cache_ttl,
        )

        client = LLMClient(config=config)
        response = client.chat(prompt, **kwargs)

        return response.content

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters for the LLM."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
