# Socratic Nexus

A universal LLM client library providing unified access to multiple AI providers with a consistent interface.

[![Tests](https://img.shields.io/badge/coverage-70.94%25-brightgreen)](https://github.com/Nireus79/Socratic-nexus)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/downloads/)

## Features

- **Multi-Provider Support**: Works with Anthropic (Claude), OpenAI (GPT-4), Google (Gemini), Ollama, and HuggingFace
- **Unified Interface**: Consistent API across all providers
- **Async/Sync Support**: Full async/await support with optional synchronous API
- **Production Ready**: 70%+ test coverage, comprehensive error handling
- **Token Tracking**: Built-in token usage monitoring and cost estimation
- **Caching**: Intelligent response caching with optional Redis support
- **Type Safe**: Full type hints with mypy support
- **No Lock-in**: Easy to switch between providers without code changes

## Installation

### Basic Installation

```bash
pip install socratic-nexus
```

### With Specific Providers

```bash
# Claude/Anthropic
pip install socratic-nexus[anthropic]

# OpenAI
pip install socratic-nexus[openai]

# Google Gemini
pip install socratic-nexus[google]

# Ollama (local LLMs)
pip install socratic-nexus[ollama]

# All providers
pip install socratic-nexus[all]
```

### With Integrations

```bash
# LangChain support
pip install socratic-nexus[langchain]

# Everything
pip install socratic-nexus[full]
```

## Quick Start

### Using Claude (Anthropic)

```python
from socratic_nexus.clients import ClaudeClient

# Initialize with API key
client = ClaudeClient(api_key="sk-ant-...")

# Generate a response
response = client.generate_response(
    "Explain quantum computing in simple terms"
)
print(response)

# Generate code
code = client.generate_code(
    "Create a Python function to calculate fibonacci numbers"
)
print(code)

# Extract insights
insights = client.extract_insights(
    "Why is photosynthesis important for Earth's ecosystem?",
    project_name="biology_study"
)
print(insights)
```

### Using OpenAI (GPT-4)

```python
from socratic_nexus.clients import OpenAIClient

client = OpenAIClient(api_key="sk-...")

response = client.generate_response(
    "What are the key principles of machine learning?"
)
print(response)
```

### Using Google Gemini

```python
from socratic_nexus.clients import GoogleClient

client = GoogleClient(api_key="...")

response = client.generate_response(
    "Explain blockchain technology"
)
print(response)
```

### Using Ollama (Local Models)

```python
from socratic_nexus.clients import OllamaClient

# Ollama runs locally, no API key needed
client = OllamaClient(base_url="http://localhost:11434")

response = client.generate_response(
    "What is the capital of France?"
)
print(response)
```

### Async Usage

```python
import asyncio
from socratic_nexus.clients import ClaudeClient

async def main():
    client = ClaudeClient(api_key="sk-ant-...")

    response = await client.generate_response_async(
        "Explain the theory of relativity"
    )
    print(response)

asyncio.run(main())
```

## Core Clients

### ClaudeClient
Anthropic's Claude model with specialized methods:
- `generate_response()` - General text generation
- `generate_code()` - Code generation
- `extract_insights()` - Extract key insights
- `generate_documentation()` - Create documentation
- `generate_business_plan()` - Business planning
- `generate_curriculum()` - Learning curriculum design
- `generate_research_protocol()` - Research methodology
- And 10+ more specialized generators

### OpenAIClient
OpenAI's GPT models with:
- Cost calculation based on token usage
- Support for GPT-4, GPT-4 Turbo, GPT-3.5
- Configurable temperature and parameters

### GoogleClient
Google's Gemini models with:
- Multi-modal support
- Advanced safety settings
- Model variant selection (pro, pro-vision)

### OllamaClient
Local LLM support for:
- Running models locally without API keys
- Custom model selection
- Development and testing without cloud costs

## Architecture

Socratic-nexus follows a layered architecture:

```
Application Code
       ↓
   Client Layer (ClaudeClient, OpenAIClient, etc.)
       ↓
   Provider Abstraction Layer
       ↓
   Provider-Specific Implementations
       ↓
   External LLM APIs
```

Each client:
- Handles provider-specific authentication
- Normalizes request/response formats
- Manages async/sync variants
- Tracks token usage and costs
- Implements caching and retry logic

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## API Reference

### Base Client Interface

All clients implement these core methods:

```python
# Generate a response to a prompt
response = client.generate_response(
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> str

# Generate code from specifications
code = client.generate_code(
    specification: str,
    language: str = "python"
) -> str

# Extract key insights from text
insights = client.extract_insights(
    text: str,
    project: Optional[ProjectContext] = None
) -> str

# Async variants
response = await client.generate_response_async(...)
code = await client.generate_code_async(...)
insights = await client.extract_insights_async(...)
```

### Token Usage Tracking

Each response includes token usage information:

```python
from socratic_nexus.clients import ClaudeClient

client = ClaudeClient(api_key="sk-ant-...")
response = client.generate_response("Your prompt here")

# Token information available in response metadata
print(f"Input tokens: {response.metadata.input_tokens}")
print(f"Output tokens: {response.metadata.output_tokens}")
print(f"Estimated cost: ${response.metadata.cost_usd}")
```

See [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for complete API documentation.

## Configuration

### Environment Variables

Set provider API keys via environment variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

Then initialize clients without explicit keys:

```python
from socratic_nexus.clients import ClaudeClient

# Reads from ANTHROPIC_API_KEY
client = ClaudeClient()
```

### Client Configuration

```python
from socratic_nexus.clients import ClaudeClient

client = ClaudeClient(
    api_key="sk-ant-...",
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    temperature=0.7,
    timeout=30
)
```

## Error Handling

```python
from socratic_nexus.clients import ClaudeClient
from anthropic import APIError

try:
    client = ClaudeClient(api_key="invalid-key")
    response = client.generate_response("Test prompt")
except APIError as e:
    print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Testing

Run the test suite:

```bash
pytest tests/
```

With coverage:

```bash
pytest --cov=socratic_nexus tests/
```

Run specific test categories:

```bash
# Unit tests only
pytest -m unit

# Integration tests (requires API keys)
pytest -m integration
```

## Performance

- **Response Caching**: Configurable in-memory caching
- **Async Support**: Non-blocking API calls for high concurrency
- **Connection Pooling**: Efficient HTTP connection management
- **Retry Logic**: Automatic retry with exponential backoff

Typical response times (with caching disabled):
- Claude: 1-5s per request
- GPT-4: 1-3s per request
- Gemini: 500ms-2s per request
- Ollama (local): 100ms-1s per request

## Security

- **API Key Protection**: Never logged or cached
- **HTTPS Only**: All communication encrypted
- **No Request Logging**: Sensitive data not persisted
- **Input Validation**: All requests validated before sending

For security issues, please see [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass and coverage remains above 70%
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **API Reference**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Quickstart Guide**: [docs/quickstart.md](docs/quickstart.md)
- **GitHub Issues**: [Report bugs](https://github.com/Nireus79/Socratic-nexus/issues)
- **Discussions**: [Ask questions](https://github.com/Nireus79/Socratic-nexus/discussions)

## Roadmap

- [ ] Support for additional providers (Cohere, Together AI, etc.)
- [ ] Function calling for all providers
- [ ] Streaming response support
- [ ] Advanced RAG integration
- [ ] Cost optimization features
- [ ] Rate limiting and quota management

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes and version history.

## Related Projects

- [Socratic Nexus Agents](https://github.com/Nireus79/Socratic-agents) - Agent framework built on socratic-nexus
- [Socratic RAG](https://github.com/Nireus79/Socratic-rag) - Retrieval-augmented generation
- [Socratic Analyzer](https://github.com/Nireus79/Socratic-analyzer) - Quality analysis and evaluation

---

Made with ❤️ for the AI community
