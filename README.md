# Socrates Nexus

**Universal LLM client for production** - Works with Claude, GPT-4, Gemini, Llama, and any LLM.

Extracted from **18 months of production use** in [Socrates AI](https://github.com/Nireus79/Socrates) platform.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/Nireus79/socrates-nexus/actions/workflows/test.yml/badge.svg)](https://github.com/Nireus79/socrates-nexus/actions)

## Why Socrates Nexus?

Most LLM clients handle the happy path. Socrates Nexus handles **production**:

- ✅ **Automatic retry logic** with exponential backoff (timeouts, rate limits, temporary errors)
- ✅ **Token usage tracking** - Know exactly what you're spending across providers
- ✅ **Streaming support** with helpers (not fighting with raw streams)
- ✅ **Async + sync APIs** - Choose what works for you
- ✅ **Multi-model fallback** - If Claude is down, try GPT-4
- ✅ **Type hints throughout** - Better IDE experience
- ✅ **Universal API** - Same code works with Claude, GPT-4, Gemini, Llama

## Quick Start

### Installation

```bash
# Install with Claude support
pip install socrates-nexus[anthropic]

# Or with all providers
pip install socrates-nexus[all]
```

### Basic Usage

```python
from socrates_nexus import LLMClient

# Create client for any LLM
client = LLMClient(
    provider="anthropic",
    model="claude-opus",
    api_key="your-api-key"
)

# Chat - automatic retries, token tracking included
response = client.chat("What is machine learning?")
print(response.content)
print(f"Cost: ${response.usage.cost_usd}")
```

### Multiple Providers (Same API)

```python
from socrates_nexus import LLMClient

# Claude
claude = LLMClient(provider="anthropic", model="claude-opus", api_key="sk-ant-...")

# GPT-4
gpt4 = LLMClient(provider="openai", model="gpt-4", api_key="sk-...")

# Gemini
gemini = LLMClient(provider="google", model="gemini-pro", api_key="...")

# Llama (local)
llama = LLMClient(provider="ollama", model="llama2", base_url="http://localhost:11434")

# All use the same API!
for client in [claude, gpt4, gemini, llama]:
    response = client.chat("Hello!")
    print(f"{client.config.provider}: {response.content}")
```

### Streaming

```python
client = LLMClient(provider="anthropic", model="claude-opus", api_key="...")

def on_chunk(chunk):
    print(chunk, end="", flush=True)

response = client.stream("Write a poem about AI", on_chunk=on_chunk)
print(f"\n\nTotal cost: ${response.usage.cost_usd}")
```

### Async

```python
import asyncio
from socrates_nexus import AsyncLLMClient

async def main():
    client = AsyncLLMClient(
        provider="anthropic",
        model="claude-opus",
        api_key="..."
    )

    # Concurrent requests
    responses = await asyncio.gather(
        client.chat("Query 1"),
        client.chat("Query 2"),
        client.chat("Query 3"),
    )

    for response in responses:
        print(response.content)

asyncio.run(main())
```

## Key Features

### 1. **Automatic Retries**

Handles transient failures automatically:
- Rate limits (HTTP 429)
- Timeout errors
- Temporary server errors (5xx)
- Exponential backoff with jitter

```python
client = LLMClient(
    provider="anthropic",
    model="claude-opus",
    api_key="...",
    retry_attempts=3,           # Number of retries
    retry_backoff_factor=2.0,   # Exponential backoff multiplier
)
```

### 2. **Token Tracking**

Track usage and costs across all providers:

```python
response = client.chat("Query")

print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Total cost: ${response.usage.cost_usd}")

# Get cumulative stats
stats = client.get_usage_stats()
print(f"Total spent: ${stats.total_cost_usd}")
```

### 3. **Multi-LLM Fallback**

Automatically try multiple models if one fails:

```python
client = LLMClient(
    provider="anthropic",
    model="claude-opus",
    api_key="...",
    fallback_providers=[
        {"provider": "openai", "model": "gpt-4", "api_key": "..."},
        {"provider": "google", "model": "gemini-pro", "api_key": "..."},
    ]
)

# If Claude fails, automatically tries GPT-4, then Gemini
response = client.chat("Query with fallback")
```

### 4. **Response Caching**

Cache identical requests to save cost and time:

```python
client = LLMClient(
    provider="anthropic",
    model="claude-opus",
    api_key="...",
    cache_responses=True,
    cache_ttl=300,  # 5 minutes
)

# First call: hits API
response1 = client.chat("What is Python?")

# Second call within 5 min: uses cache (instant)
response2 = client.chat("What is Python?")

print(f"Saved: ${response1.usage.cost_usd * 0.9}")
```

## Supported Providers

| Provider | Models | Status |
|----------|--------|--------|
| Anthropic | Claude 3 (Opus, Sonnet, Haiku) | ✅ In development |
| OpenAI | GPT-4, GPT-3.5-turbo | ✅ In development |
| Google | Gemini Pro, Gemini Vision | ✅ In development |
| Ollama | Llama 2, Mistral, etc. | ✅ In development |
| HuggingFace | Any open-source model | ✅ In development |

## Examples

See the `examples/` directory for complete examples:

- `01_anthropic_claude.py` - Using Claude
- `02_openai_gpt4.py` - Using GPT-4
- `03_google_gemini.py` - Using Gemini
- `04_ollama_local.py` - Using local Llama
- `05_streaming.py` - Streaming responses
- `06_async_calls.py` - Async usage
- `07_token_tracking.py` - Token tracking and costs
- `08_multi_model_fallback.py` - Multi-model fallback
- `09_error_handling.py` - Error handling

## Documentation

- [Quick Start](docs/quickstart.md) - Get started in 5 minutes
- [Providers Guide](docs/providers.md) - Setup for each LLM provider
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Advanced Usage](docs/advanced.md) - Caching, fallbacks, monitoring
- [Comparisons](docs/comparisons.md) - vs raw SDKs

## Development

### Setup

```bash
# Clone repo
git clone https://github.com/Nireus79/socrates-nexus.git
cd socrates-nexus

# Install dev dependencies
pip install -e ".[dev,all]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
ruff check src/ tests/
```

### Testing

```bash
# All tests
pytest tests/ -v

# Only fast tests
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ --cov=socrates_nexus --cov-report=html
```

## Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE)

## Origins

Socrates Nexus is extracted from [Socrates AI](https://github.com/Nireus79/Socrates), a collaborative AI platform. It's battle-tested in production and used for orchestrating multiple LLMs.

## Roadmap

- ✅ Base client structure
- 🔄 Provider implementations (Claude, GPT-4, Gemini, Ollama, HuggingFace)
- 🔄 Streaming for all providers
- ⏳ Vision models support
- ⏳ Function calling for all providers
- ⏳ Batch processing
- ⏳ Monitoring and observability

## Support

- **Issues**: [GitHub Issues](https://github.com/Nireus79/socrates-nexus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Nireus79/socrates-nexus/discussions)
- **Sponsor**: [GitHub Sponsors](https://github.com/sponsors/Nireus79)

---

**Made with ❤️ as part of the Socrates ecosystem**
