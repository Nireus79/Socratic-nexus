"""Client integrations for Socrates AI"""

from .claude_client import ClaudeClient

__all__ = ["ClaudeClient"]

# Optional client imports with graceful fallback
try:
    from .openai_client import OpenAIClient  # noqa: F401
    __all__.append("OpenAIClient")
except ImportError:
    pass

try:
    from .google_client import GoogleClient  # noqa: F401
    __all__.append("GoogleClient")
except ImportError:
    pass

try:
    from .ollama_client import OllamaClient  # noqa: F401
    __all__.append("OllamaClient")
except ImportError:
    pass
