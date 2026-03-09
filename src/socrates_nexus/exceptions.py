"""Exception classes for Socrates Nexus."""


class LLMError(Exception):
    """Base exception for Socrates Nexus."""

    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int = None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(f"{message}. Retry after {retry_after} seconds." if retry_after else message)


class InvalidAPIKeyError(LLMError):
    """Raised when API key is invalid."""

    pass


class ContextLengthExceededError(LLMError):
    """Raised when input exceeds model's context length."""

    pass


class ModelNotFoundError(LLMError):
    """Raised when model is not found."""

    pass


class ProviderError(LLMError):
    """Raised when provider encounters an error."""

    pass


class StreamingError(LLMError):
    """Raised when streaming fails."""

    pass


class ConfigurationError(LLMError):
    """Raised when configuration is invalid."""

    pass
