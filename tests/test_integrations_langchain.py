"""
Tests for LangChain Integration.

Tests for SocratesNexusLLM - LangChain integration with Socrates Nexus.
"""

from unittest.mock import Mock, patch
from socratic_nexus.integrations.langchain import SocratesNexusLLM
from socratic_nexus import ChatResponse, TokenUsage


class TestSocratesNexusLLMInitialization:
    """Test SocratesNexusLLM initialization."""

    def test_initialization_defaults(self):
        """Test LLM initializes with default parameters."""
        llm = SocratesNexusLLM()
        assert llm.provider == "anthropic"
        assert llm.model == "claude-opus"
        assert llm.temperature == 0.7
        assert llm.max_tokens is None
        assert llm.retry_attempts == 3
        assert llm.cache_responses is True

    def test_initialization_custom_provider(self):
        """Test LLM initializes with custom provider."""
        llm = SocratesNexusLLM(provider="openai", model="gpt-4")
        assert llm.provider == "openai"
        assert llm.model == "gpt-4"

    def test_initialization_custom_parameters(self):
        """Test LLM initializes with custom parameters."""
        llm = SocratesNexusLLM(
            provider="google",
            model="gemini-pro",
            temperature=0.5,
            max_tokens=512,
            retry_attempts=5,
            cache_responses=False,
        )
        assert llm.provider == "google"
        assert llm.model == "gemini-pro"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 512
        assert llm.retry_attempts == 5
        assert llm.cache_responses is False

    def test_initialization_with_api_key(self):
        """Test LLM initializes with API key."""
        llm = SocratesNexusLLM(api_key="test-key-123")
        assert llm.api_key == "test-key-123"


class TestSocratesNexusLLMCall:
    """Test _call method (core LangChain interface)."""

    def test_call_basic(self):
        """Test basic _call execution."""
        llm = SocratesNexusLLM()
        with patch("socratic_nexus.integrations.langchain.llm.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = ChatResponse(
                content="Test response",
                provider="anthropic",
                model="claude-opus",
                usage=TokenUsage(
                    provider="anthropic",
                    model="claude-opus",
                    input_tokens=10,
                    output_tokens=20,
                    total_tokens=30,
                ),
            )
            mock_client.chat.return_value = mock_response

            result = llm._call("What is AI?")

            assert result == "Test response"
            mock_client.chat.assert_called_once()

    def test_call_with_stop_sequences(self):
        """Test _call with stop sequences."""
        llm = SocratesNexusLLM()
        with patch("socratic_nexus.integrations.langchain.llm.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = ChatResponse(
                content="Response",
                provider="anthropic",
                model="claude-opus",
                usage=TokenUsage(
                    provider="anthropic",
                    model="claude-opus",
                    input_tokens=5,
                    output_tokens=10,
                    total_tokens=15,
                ),
            )

            result = llm._call("Query", stop=["END"])

            assert result == "Response"

    def test_call_with_kwargs(self):
        """Test _call with additional kwargs."""
        llm = SocratesNexusLLM()
        with patch("socratic_nexus.integrations.langchain.llm.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = ChatResponse(
                content="Response",
                provider="anthropic",
                model="claude-opus",
                usage=TokenUsage(
                    provider="anthropic",
                    model="claude-opus",
                    input_tokens=5,
                    output_tokens=10,
                    total_tokens=15,
                ),
            )

            result = llm._call("Query", temperature=0.5, max_tokens=100)

            assert result == "Response"

    def test_call_creates_client_with_config(self):
        """Test that _call creates client with correct config."""
        llm = SocratesNexusLLM(
            provider="openai",
            model="gpt-4",
            temperature=0.9,
            max_tokens=2048,
            api_key="test-key",
        )
        with patch("socratic_nexus.integrations.langchain.llm.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = ChatResponse(
                content="Response",
                provider="openai",
                model="gpt-4",
                usage=TokenUsage(
                    provider="openai",
                    model="gpt-4",
                    input_tokens=5,
                    output_tokens=10,
                    total_tokens=15,
                ),
            )

            llm._call("Query")

            # Verify LLMClient was called with correct config
            assert mock_client_class.called


class TestSocratesNexusLLMProperties:
    """Test LLM properties and metadata."""

    def test_llm_type(self):
        """Test _llm_type property."""
        llm = SocratesNexusLLM()
        assert llm._llm_type == "socratic_nexus"

    def test_identifying_params(self):
        """Test _identifying_params property."""
        llm = SocratesNexusLLM(
            provider="anthropic", model="claude-opus", temperature=0.7, max_tokens=512
        )
        params = llm._identifying_params

        assert params["provider"] == "anthropic"
        assert params["model"] == "claude-opus"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 512

    def test_identifying_params_with_none_max_tokens(self):
        """Test _identifying_params when max_tokens is None."""
        llm = SocratesNexusLLM(max_tokens=None)
        params = llm._identifying_params

        assert params["max_tokens"] is None


class TestSocratesNexusLLMProviderSupport:
    """Test support for different providers."""

    def test_anthropic_provider(self):
        """Test Anthropic provider configuration."""
        llm = SocratesNexusLLM(provider="anthropic", model="claude-opus")
        assert llm.provider == "anthropic"
        assert llm.model == "claude-opus"

    def test_openai_provider(self):
        """Test OpenAI provider configuration."""
        llm = SocratesNexusLLM(provider="openai", model="gpt-4")
        assert llm.provider == "openai"
        assert llm.model == "gpt-4"

    def test_google_provider(self):
        """Test Google provider configuration."""
        llm = SocratesNexusLLM(provider="google", model="gemini-pro")
        assert llm.provider == "google"
        assert llm.model == "gemini-pro"

    def test_ollama_provider(self):
        """Test Ollama provider configuration."""
        llm = SocratesNexusLLM(provider="ollama", model="llama2")
        assert llm.provider == "ollama"
        assert llm.model == "llama2"


class TestSocratesNexusLLMTemperature:
    """Test temperature settings."""

    def test_temperature_default(self):
        """Test default temperature."""
        llm = SocratesNexusLLM()
        assert llm.temperature == 0.7

    def test_temperature_zero(self):
        """Test zero temperature (deterministic)."""
        llm = SocratesNexusLLM(temperature=0.0)
        assert llm.temperature == 0.0

    def test_temperature_high(self):
        """Test high temperature (creative)."""
        llm = SocratesNexusLLM(temperature=1.5)
        assert llm.temperature == 1.5

    def test_temperature_in_identifying_params(self):
        """Test that temperature appears in identifying params."""
        llm = SocratesNexusLLM(temperature=0.5)
        params = llm._identifying_params
        assert params["temperature"] == 0.5


class TestSocratesNexusLLMMaxTokens:
    """Test max_tokens configuration."""

    def test_max_tokens_default(self):
        """Test default max_tokens."""
        llm = SocratesNexusLLM()
        assert llm.max_tokens is None

    def test_max_tokens_set(self):
        """Test setting max_tokens."""
        llm = SocratesNexusLLM(max_tokens=1024)
        assert llm.max_tokens == 1024

    def test_max_tokens_large_value(self):
        """Test large max_tokens value."""
        llm = SocratesNexusLLM(max_tokens=4096)
        assert llm.max_tokens == 4096

    def test_max_tokens_in_identifying_params(self):
        """Test that max_tokens appears in identifying params."""
        llm = SocratesNexusLLM(max_tokens=512)
        params = llm._identifying_params
        assert params["max_tokens"] == 512


class TestSocratesNexusLLMRetryConfiguration:
    """Test retry configuration."""

    def test_retry_attempts_default(self):
        """Test default retry attempts."""
        llm = SocratesNexusLLM()
        assert llm.retry_attempts == 3

    def test_retry_attempts_custom(self):
        """Test custom retry attempts."""
        llm = SocratesNexusLLM(retry_attempts=5)
        assert llm.retry_attempts == 5

    def test_retry_attempts_one(self):
        """Test retry_attempts set to 1."""
        llm = SocratesNexusLLM(retry_attempts=1)
        assert llm.retry_attempts == 1


class TestSocratesNexusLLMCacheConfiguration:
    """Test cache configuration."""

    def test_cache_enabled_by_default(self):
        """Test that cache is enabled by default."""
        llm = SocratesNexusLLM()
        assert llm.cache_responses is True

    def test_cache_disabled(self):
        """Test disabling cache."""
        llm = SocratesNexusLLM(cache_responses=False)
        assert llm.cache_responses is False

    def test_cache_ttl_default(self):
        """Test default cache TTL."""
        llm = SocratesNexusLLM()
        assert llm.cache_ttl == 300

    def test_cache_ttl_custom(self):
        """Test custom cache TTL."""
        llm = SocratesNexusLLM(cache_ttl=600)
        assert llm.cache_ttl == 600


class TestSocratesNexusLLMCallbackManager:
    """Test callback manager integration."""

    def test_call_with_run_manager(self):
        """Test _call with run_manager parameter."""
        llm = SocratesNexusLLM()
        run_manager = Mock()

        with patch("socratic_nexus.integrations.langchain.llm.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = ChatResponse(
                content="Response",
                provider="anthropic",
                model="claude-opus",
                usage=TokenUsage(
                    provider="anthropic",
                    model="claude-opus",
                    input_tokens=5,
                    output_tokens=10,
                    total_tokens=15,
                ),
            )

            result = llm._call("Query", run_manager=run_manager)

            assert result == "Response"


class TestSocratesNexusLLMMultiProviderUsage:
    """Test multi-provider usage scenarios."""

    def test_switching_providers_in_langchain_chain(self):
        """Test switching providers for use in LangChain chains."""
        # Create initial LLM
        llm1 = SocratesNexusLLM(provider="anthropic", model="claude-opus")
        assert llm1.provider == "anthropic"

        # Create alternative LLM with different provider
        llm2 = SocratesNexusLLM(provider="openai", model="gpt-4")
        assert llm2.provider == "openai"

        # Both should be usable
        assert llm1._llm_type == "socratic_nexus"
        assert llm2._llm_type == "socratic_nexus"

    def test_multiple_llm_instances(self):
        """Test creating multiple LLM instances with different configs."""
        llms = [
            SocratesNexusLLM(provider="anthropic", model="claude-opus"),
            SocratesNexusLLM(provider="openai", model="gpt-4"),
            SocratesNexusLLM(provider="google", model="gemini-pro"),
        ]

        assert len(llms) == 3
        assert llms[0].provider == "anthropic"
        assert llms[1].provider == "openai"
        assert llms[2].provider == "google"


class TestSocratesNexusLLMEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prompt(self):
        """Test handling of empty prompt."""
        llm = SocratesNexusLLM()
        with patch("socratic_nexus.integrations.langchain.llm.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = ChatResponse(
                content="Response",
                provider="anthropic",
                model="claude-opus",
                usage=TokenUsage(
                    provider="anthropic",
                    model="claude-opus",
                    input_tokens=0,
                    output_tokens=5,
                    total_tokens=5,
                ),
            )

            result = llm._call("")
            assert result == "Response"

    def test_very_long_prompt(self):
        """Test handling of very long prompt."""
        llm = SocratesNexusLLM()
        long_prompt = "A" * 10000

        with patch("socratic_nexus.integrations.langchain.llm.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = ChatResponse(
                content="Response",
                provider="anthropic",
                model="claude-opus",
                usage=TokenUsage(
                    provider="anthropic",
                    model="claude-opus",
                    input_tokens=5000,
                    output_tokens=5,
                    total_tokens=5005,
                ),
            )

            result = llm._call(long_prompt)
            assert result == "Response"

    def test_special_characters_in_prompt(self):
        """Test handling of special characters."""
        llm = SocratesNexusLLM()
        special_prompt = "What is \\n\\t\\r 日本語 emoji 😀?"

        with patch("socratic_nexus.integrations.langchain.llm.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = ChatResponse(
                content="Response",
                provider="anthropic",
                model="claude-opus",
                usage=TokenUsage(
                    provider="anthropic",
                    model="claude-opus",
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                ),
            )

            result = llm._call(special_prompt)
            assert result == "Response"
