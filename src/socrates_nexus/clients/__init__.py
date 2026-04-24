"""Client integrations for Socrates AI"""

from .claude_client import ClaudeClient

# Optional client imports with graceful fallback
_available_clients = ["ClaudeClient"]

try:
    from .openai_client import OpenAIClient
    _available_clients.append("OpenAIClient")
except ImportError:
    pass

try:
    from .google_client import GoogleClient
    _available_clients.append("GoogleClient")
except ImportError:
    pass

try:
    from .ollama_client import OllamaClient
    _available_clients.append("OllamaClient")
except ImportError:
    pass

__all__ = _available_clients
