# Socrates Nexus - API Reference

## LLMClient

Main synchronous client for LLM interactions.

### Constructor

```python
LLMClient(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    retry_attempts: int = 3,
    retry_backoff_factor: float = 2.0,
    request_timeout: int = 60,
    cache_responses: bool = True,
    cache_ttl: int = 300
)
```

### Methods

#### chat(message, **kwargs) -> Response

Send a chat message and get response.

```python
response = client.chat("What is Python?")
print(response.content)  # Generated text
print(response.usage.cost_usd)  # Cost in USD
```

#### stream(message, on_chunk=None) -> Response

Stream response chunks.

```python
def on_chunk(chunk):
    print(chunk, end="", flush=True)

response = client.stream("Long query", on_chunk=on_chunk)
```

#### get_usage_stats() -> Dict

Get cumulative usage statistics.

```python
stats = client.get_usage_stats()
print(f"Total cost: ${stats['total_cost_usd']}")
print(f"Total requests: {stats['total_requests']}")
```

#### add_usage_callback(callback)

Register callback for usage tracking.

```python
def track_usage(usage):
    print(f"Used {usage.total_tokens} tokens, ${usage.cost_usd}")

client.add_usage_callback(track_usage)
```

---

## AsyncLLMClient

Asynchronous version of LLMClient.

### async chat(message, **kwargs) -> Response

```python
response = await client.chat("What is Python?")
```

### async stream(message, on_chunk=None) -> Response

```python
async def on_chunk(chunk):
    print(chunk, end="", flush=True)

response = await client.stream("Query", on_chunk=on_chunk)
```

### Concurrent Requests

```python
import asyncio

responses = await asyncio.gather(
    client.chat("Query 1"),
    client.chat("Query 2"),
    client.chat("Query 3")
)
```

---

## LLMConfig

Configuration dataclass.

```python
@dataclass
class LLMConfig:
    provider: str                    # Required
    model: str                       # Required
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    request_timeout: int = 60
    cache_responses: bool = True
    cache_ttl: int = 300
```

---

## Response

Response dataclass returned by LLM calls.

```python
@dataclass
class Response:
    content: str                     # Generated text
    stop_reason: str                 # Why generation stopped
    usage: TokenUsage               # Token usage info
    model: str                       # Model used
    provider: str                    # Provider used
    timestamp: datetime              # Response time
```

---

## TokenUsage

Token usage and cost information.

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
```

---

## Exceptions

### NexusError

Base exception for all Socrates Nexus errors.

```python
try:
    response = client.chat("query")
except NexusError as e:
    print(f"Error: {e.message}")
    print(f"Code: {e.error_code}")
```

### RateLimitError

Raised when rate limit is exceeded (HTTP 429).

```python
except RateLimitError as e:
    print(f"Retry after: {e.retry_after} seconds")
```

### AuthenticationError

Raised when authentication fails (HTTP 401).

```python
except AuthenticationError:
    # Refresh API key
    pass
```

### InvalidAPIKeyError

Raised when API key is invalid.

### TimeoutError

Raised when request times out.

### ContextLengthExceededError

Raised when input exceeds max tokens.

```python
except ContextLengthExceededError as e:
    # Truncate message and retry
    truncated = message[:e.max_tokens]
    response = client.chat(truncated)
```

### ModelNotFoundError

Raised when model doesn't exist.

---

## Providers

### AnthropicProvider

Claude models from Anthropic.

Models:
- claude-opus
- claude-3-sonnet
- claude-3-haiku
- claude-3-5-sonnet

### OpenAIProvider

OpenAI models.

Models:
- gpt-4
- gpt-4o
- gpt-3.5-turbo

### GoogleProvider

Google Gemini models.

Models:
- gemini-1.5-pro
- gemini-1.5-flash

### OllamaProvider

Local Ollama models.

Models:
- llama2
- mistral
- neural-chat
- orca

---

## Best Practices

### 1. Error Handling

```python
from socrates_nexus import (
    LLMClient,
    RateLimitError,
    AuthenticationError,
    NexusError
)

try:
    response = client.chat("query")
except RateLimitError:
    # Wait and retry
    pass
except AuthenticationError:
    # Refresh credentials
    pass
except NexusError as e:
    # Log and handle
    logger.error(f"LLM error: {e.message}")
```

### 2. Cost Management

```python
# Track costs
response = client.chat("query")
if response.usage.cost_usd > 0.01:
    alert("Expensive request")

# Batch similar requests
for query in queries:
    response = client.chat(query)
```

### 3. Provider Selection

```python
# Fast/cheap: GPT-3.5
# Balanced: Claude Sonnet, GPT-4
# Best: Claude Opus, GPT-4
```

### 4. Streaming for Long Responses

```python
def on_chunk(chunk):
    send_to_user(chunk)  # Stream to user

response = client.stream("Long query", on_chunk=on_chunk)
```
