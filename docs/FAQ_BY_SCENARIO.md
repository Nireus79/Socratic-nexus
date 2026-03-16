# Socrates Nexus - FAQ by Scenario

## Scenario 1: Comparing LLM Providers

How do I compare responses from different LLM providers?

Answer: Use the same query with different providers and compare results.

Example:
```python
from socrates_nexus import LLMClient

query = "Explain quantum computing"
for provider in ["anthropic", "openai", "google"]:
    client = LLMClient(provider=provider, model="...")
    response = client.chat(query)
    print(f"{provider}: ${response.usage.cost_usd}")
```

## Scenario 2: Managing Costs

How do I monitor LLM costs in production?

Answer: Use callbacks to track usage and log costs.

Example:
```python
client = LLMClient(provider="anthropic", model="claude-opus")

def log_cost(usage):
    print(f"Cost: ${usage.cost_usd}")

client.add_usage_callback(log_cost)
```

## Scenario 3: Handling Rate Limits

How do I handle rate limiting gracefully?

Answer: Catch RateLimitError and implement backoff.

Example:
```python
from socrates_nexus import RateLimitError
import time

try:
    response = client.chat("query")
except RateLimitError as e:
    time.sleep(e.retry_after)
    response = client.chat("query")
```

## Scenario 4: Streaming Responses

How do I stream long responses in real-time?

Answer: Use stream() method with callback.

Example:
```python
def on_chunk(chunk):
    print(chunk, end="", flush=True)

response = client.stream("Long query", on_chunk=on_chunk)
```

## Scenario 5: Provider Fallback

How do I implement automatic provider fallback?

Answer: Try providers in sequence until one succeeds.

Example:
```python
providers = [
    {"provider": "anthropic", "model": "claude-opus"},
    {"provider": "openai", "model": "gpt-4"},
    {"provider": "google", "model": "gemini-pro"},
]

for config in providers:
    try:
        client = LLMClient(**config)
        return client.chat(query)
    except Exception:
        continue
```

## Scenario 6: Batch Processing

How do I process multiple queries efficiently?

Answer: Use AsyncLLMClient for concurrent requests.

Example:
```python
import asyncio
from socrates_nexus import AsyncLLMClient

async def batch():
    client = AsyncLLMClient(provider="anthropic")
    results = await asyncio.gather(
        *[client.chat(q) for q in queries]
    )
    return results

asyncio.run(batch())
```

## Scenario 7: Caching for Development

How do I reduce costs during development?

Answer: Enable response caching.

Example:
```python
dev_client = LLMClient(
    provider="anthropic",
    cache_responses=True,
    cache_ttl=3600
)
```

## Common Errors

### Invalid API Key
Use correct API key. Check environment variables.

### Context Length Exceeded  
Truncate long messages to fit model limits.

### Model Not Found
Use correct model name for provider.
