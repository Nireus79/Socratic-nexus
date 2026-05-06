# Advanced Usage Guide

Advanced patterns and techniques for production use of Socrates Nexus.

## Table of Contents

- [Response Caching](#response-caching)
- [Multi-Provider Fallback](#multi-provider-fallback)
- [Custom Callbacks](#custom-callbacks)
- [Performance Optimization](#performance-optimization)
- [Error Recovery](#error-recovery)
- [Cost Monitoring](#cost-monitoring)
- [Streaming Best Practices](#streaming-best-practices)
- [Concurrency Patterns](#concurrency-patterns)
- [Encryption & Security](./ENCRYPTION.md)

---

## Response Caching

### TTL-Based Caching

Cache identical requests to reduce costs and latency:

```python
from socrates_nexus import LLMClient, LLMConfig

config = LLMConfig(
    provider="anthropic",
    model="claude-opus",
    api_key="...",
    cache_responses=True,
    cache_ttl=300,  # 5 minutes
)

client = LLMClient(config=config)

# First call: hits API (slow, costs money)
response1 = client.chat("What is Python?")
print(f"Cost: ${response1.usage.cost_usd:.6f}")

# Second call (within 5 min): uses cache (instant, free)
response2 = client.chat("What is Python?")
print(f"Cost: ${response2.usage.cost_usd:.6f}")  # $0.000000

# Check cache statistics
stats = client.get_usage_stats()
print(f"Total requests: {stats.total_requests}")
print(f"Total cost: ${stats.total_cost_usd:.6f}")
```

### Cache Key Strategy

Cache keys are based on message content (SHA256 hash):

```python
# These are cached separately (different keys):
client.chat("What is Python?")
client.chat("What is Python? ")  # Note: space at end - different key
client.chat("what is python?")    # Different case - different key

# These use the same cache:
client.chat("What is Python?")
client.chat("What is Python?")  # Exact same - cache hit
```

### Manual Cache Management

```python
# Clear cache when needed
client.clear_cache()

# Disable caching for sensitive queries
config_no_cache = LLMConfig(
    provider="anthropic",
    model="claude-opus",
    api_key="...",
    cache_responses=False,  # Disable
)

client_no_cache = LLMClient(config=config_no_cache)
```

---

## Multi-Provider Fallback

### Sequential Fallback Pattern

Try providers in order, fail over on error:

```python
from socrates_nexus import LLMClient, LLMError

def chat_with_fallback(message: str) -> str:
    """Try Claude first, then GPT-4, then Gemini."""

    providers = [
        {
            "provider": "anthropic",
            "model": "claude-opus",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        },
        {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        {
            "provider": "google",
            "model": "gemini-1.5-pro",
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
    ]

    for config in providers:
        try:
            print(f"Trying {config['provider']}...")
            client = LLMClient(**config)
            response = client.chat(message)
            print(f"Success with {config['provider']}")
            return response.content
        except LLMError as e:
            print(f"Failed: {e.message}")
            continue

    raise Exception("All providers failed")

# Usage
result = chat_with_fallback("Explain machine learning")
print(result)
```

### Parallel Fallback Pattern

Try all providers concurrently, use first successful:

```python
import asyncio
from socrates_nexus import AsyncLLMClient

async def chat_parallel_fallback(message: str) -> str:
    """Try all providers in parallel."""

    clients = [
        AsyncLLMClient(
            provider="anthropic",
            model="claude-opus",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        ),
        AsyncLLMClient(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        AsyncLLMClient(
            provider="google",
            model="gemini-1.5-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
        ),
    ]

    # Launch all requests in parallel
    tasks = [client.chat(message) for client in clients]

    # Return first successful response
    for coro in asyncio.as_completed(tasks):
        try:
            response = await coro
            return response.content
        except Exception:
            continue

    raise Exception("All providers failed")

# Usage
result = asyncio.run(chat_parallel_fallback("Explain machine learning"))
print(result)
```

### Cost-Optimized Fallback

Try cheaper models first, escalate if needed:

```python
from socrates_nexus import LLMClient

def chat_cost_optimized(message: str, budget: float = 0.01) -> str:
    """Try cheap models first, escalate within budget."""

    # Ordered by cost (cheapest first)
    providers = [
        ("anthropic", "claude-haiku-4-5-20251001"),      # ~$0.001
        ("openai", "gpt-3.5-turbo"),                      # ~$0.005
        ("anthropic", "claude-3-5-sonnet-20241022"),     # ~$0.015
        ("openai", "gpt-4"),                              # ~$0.03
    ]

    for provider, model in providers:
        try:
            client = LLMClient(provider=provider, model=model, api_key="...")
            response = client.chat(message)

            if response.usage.cost_usd <= budget:
                return response.content
            else:
                print(f"Cost ${response.usage.cost_usd:.6f} exceeds budget ${budget}")
                continue
        except Exception as e:
            print(f"Failed: {e}")
            continue

    raise Exception("No provider within budget")
```

---

## Custom Callbacks

### Track Usage in Real-Time

```python
from socrates_nexus import LLMClient, TokenUsage

client = LLMClient(provider="anthropic", model="claude-opus", api_key="...")

# Track expensive requests
def alert_on_expensive(usage: TokenUsage):
    if usage.cost_usd > 0.01:
        print(f"ALERT: Expensive request! ${usage.cost_usd:.6f}")
        print(f"  Provider: {usage.provider}")
        print(f"  Tokens: {usage.total_tokens}")

# Track by provider
provider_costs = {}

def track_by_provider(usage: TokenUsage):
    provider = usage.provider
    provider_costs[provider] = provider_costs.get(provider, 0) + usage.cost_usd
    print(f"{provider} total: ${provider_costs[provider]:.6f}")

# Register callbacks
client.add_usage_callback(alert_on_expensive)
client.add_usage_callback(track_by_provider)

# Make requests - callbacks fire automatically
client.chat("Query 1")
client.chat("Query 2")
client.chat("Query 3")
```

### Logging Callback

```python
import json
from datetime import datetime
from socrates_nexus import TokenUsage

def log_to_file(usage: TokenUsage):
    """Log all requests to JSON file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "provider": usage.provider,
        "model": usage.model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": usage.cost_usd,
        "latency_ms": usage.latency_ms,
    }

    with open("requests.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

client.add_usage_callback(log_to_file)
```

---

## Performance Optimization

### Async Concurrent Requests

Process multiple requests in parallel:

```python
import asyncio
from socrates_nexus import AsyncLLMClient

async def process_batch(queries: list) -> list:
    """Process multiple queries concurrently."""

    client = AsyncLLMClient(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",  # Faster/cheaper
        api_key="...",
    )

    # Launch all requests in parallel
    tasks = [client.chat(query) for query in queries]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    return [r.content for r in responses]

# Usage
queries = [
    "What is AI?",
    "What is ML?",
    "What is DL?",
    "What is NLP?",
    "What is CV?",
]

results = asyncio.run(process_batch(queries))
for query, result in zip(queries, results):
    print(f"Q: {query}")
    print(f"A: {result[:100]}...\n")
```

### Batch Processing

Process items in batches for better throughput:

```python
async def process_in_batches(items: list, batch_size: int = 5):
    """Process items in batches."""

    client = AsyncLLMClient(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="...")

    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        print(f"Processing batch {i // batch_size + 1} ({len(batch)} items)...")

        tasks = [client.chat(f"Analyze: {item}") for item in batch]
        batch_results = await asyncio.gather(*tasks)

        results.extend([r.content for r in batch_results])

        # Optional: pause between batches to avoid rate limits
        await asyncio.sleep(1)

    return results
```

### Model Selection by Latency

Choose model based on latency requirements:

```python
def get_fast_response(message: str) -> str:
    """Use fastest model for low-latency requirements."""
    # Haiku is typically fastest
    client = LLMClient(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        api_key="...",
    )
    return client.chat(message).content

def get_best_response(message: str) -> str:
    """Use best model for accuracy."""
    # Opus is typically best
    client = LLMClient(
        provider="anthropic",
        model="claude-opus",
        api_key="...",
    )
    return client.chat(message).content

def get_balanced_response(message: str) -> str:
    """Use balanced model for most use cases."""
    # Sonnet is good balance
    client = LLMClient(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key="...",
    )
    return client.chat(message).content
```

---

## Error Recovery

### Retry with Exponential Backoff

```python
from socrates_nexus import LLMClient, LLMConfig, RateLimitError

config = LLMConfig(
    provider="anthropic",
    model="claude-opus",
    api_key="...",
    retry_attempts=5,              # More aggressive retries
    retry_backoff_factor=2.0,      # Exponential backoff
    request_timeout=120,           # Longer timeout
)

client = LLMClient(config=config)

try:
    response = client.chat("Complex query requiring multiple retries")
    print(response.content)
except Exception as e:
    print(f"Failed after all retries: {e}")
```

### Smart Error Handling

```python
from socrates_nexus import (
    LLMError,
    RateLimitError,
    ContextLengthExceededError,
    AuthenticationError,
)

def handle_error(error: Exception, message: str) -> str:
    """Handle different error types appropriately."""

    if isinstance(error, RateLimitError):
        if error.retry_after:
            print(f"Rate limited. Wait {error.retry_after}s before retrying")
        return "RATE_LIMITED"

    elif isinstance(error, ContextLengthExceededError):
        # Try with shorter input
        print("Input too long, truncating...")
        shorter_message = message[:500]
        return shorter_message

    elif isinstance(error, AuthenticationError):
        print("Check your API key")
        return "AUTH_ERROR"

    elif isinstance(error, LLMError):
        print(f"LLM Error ({error.error_code}): {error.message}")
        return "LLM_ERROR"

    else:
        print(f"Unexpected error: {error}")
        return "UNKNOWN_ERROR"

# Usage
try:
    response = client.chat(very_long_message)
except Exception as e:
    result = handle_error(e, very_long_message)
```

---

## Cost Monitoring

### Budget Tracking

```python
from socrates_nexus import LLMClient

class BudgetTracker:
    def __init__(self, budget: float):
        self.budget = budget
        self.spent = 0
        self.requests = 0

    def track_usage(self, usage):
        self.spent += usage.cost_usd
        self.requests += 1
        remaining = self.budget - self.spent

        print(f"Request #{self.requests}")
        print(f"  Cost: ${usage.cost_usd:.6f}")
        print(f"  Total spent: ${self.spent:.6f}")
        print(f"  Remaining budget: ${remaining:.6f}")

        if remaining < 0:
            print("⚠️ BUDGET EXCEEDED!")
        elif remaining < self.budget * 0.1:
            print("⚠️ LOW BUDGET (< 10% remaining)")

# Usage
tracker = BudgetTracker(budget=1.00)  # $1 budget

client = LLMClient(provider="anthropic", model="claude-opus", api_key="...")
client.add_usage_callback(tracker.track_usage)

for i in range(10):
    client.chat(f"Query {i}")
```

### Cost Analysis by Provider

```python
def analyze_costs(client: LLMClient):
    """Analyze costs across providers."""

    stats = client.get_usage_stats()

    print("Cost Breakdown by Provider:")
    print("-" * 50)

    for provider, p_stats in stats.by_provider.items():
        requests = p_stats['requests']
        cost = p_stats['cost_usd']
        tokens = p_stats.get('total_tokens', 0)
        avg_cost = cost / requests if requests > 0 else 0
        cost_per_1k = (cost / tokens * 1000) if tokens > 0 else 0

        print(f"\n{provider.upper()}")
        print(f"  Requests: {requests}")
        print(f"  Total Cost: ${cost:.6f}")
        print(f"  Avg Cost/Request: ${avg_cost:.6f}")
        print(f"  Cost per 1K tokens: ${cost_per_1k:.6f}")
        print(f"  Total Tokens: {tokens}")
```

---

## Streaming Best Practices

### Progress Tracking

```python
import time

def stream_with_progress(client, message: str):
    """Stream with progress updates."""

    chunks = []
    start_time = time.time()
    last_update = start_time

    def on_chunk(chunk: str):
        nonlocal last_update
        chunks.append(chunk)

        # Update progress every 0.5 seconds
        now = time.time()
        if now - last_update > 0.5:
            elapsed = now - start_time
            chars = sum(len(c) for c in chunks)
            rate = chars / elapsed if elapsed > 0 else 0

            print(f"\r[{elapsed:.1f}s] {chars} chars @ {rate:.0f} chars/s", end="", flush=True)
            last_update = now

    response = client.stream(message, on_chunk=on_chunk)

    elapsed = time.time() - start_time
    total_chars = sum(len(c) for c in chunks)
    final_rate = total_chars / elapsed if elapsed > 0 else 0

    print(f"\n✓ Complete: {total_chars} chars in {elapsed:.1f}s ({final_rate:.0f} chars/s)")
    print(f"Cost: ${response.usage.cost_usd:.6f}")

    return "".join(chunks)
```

### Buffered Streaming

```python
def stream_with_buffer(client, message: str, buffer_size: int = 5):
    """Buffer chunks before processing."""

    buffer = []
    processed_text = []

    def on_chunk(chunk: str):
        buffer.append(chunk)

        if len(buffer) >= buffer_size:
            # Process buffer
            text = "".join(buffer)
            processed_text.append(process_text(text))
            buffer.clear()

    response = client.stream(message, on_chunk=on_chunk)

    # Process remaining buffer
    if buffer:
        text = "".join(buffer)
        processed_text.append(process_text(text))

    return "".join(processed_text)

def process_text(text: str) -> str:
    """Custom text processing."""
    return text.upper()  # Example: uppercase
```

---

## Concurrency Patterns

### Worker Pool Pattern

```python
import asyncio
from socrates_nexus import AsyncLLMClient

async def worker_pool(tasks: list, num_workers: int = 3):
    """Process tasks with a pool of workers."""

    queue = asyncio.Queue()
    results = []

    # Add all tasks to queue
    for task in tasks:
        await queue.put(task)

    async def worker(worker_id: int):
        client = AsyncLLMClient(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="...")

        while not queue.empty():
            task = await queue.get()
            try:
                print(f"Worker {worker_id} processing: {task[:50]}...")
                response = await client.chat(task)
                results.append(response.content)
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
            finally:
                queue.task_done()

    # Create and run workers
    workers = [worker(i) for i in range(num_workers)]
    await asyncio.gather(*workers)

    return results

# Usage
tasks = [f"Explain concept {i}" for i in range(20)]
results = asyncio.run(worker_pool(tasks, num_workers=3))
```

---

## Best Practices Summary

1. **Caching**: Enable for repeated queries to save costs
2. **Fallback**: Always have provider fallback for reliability
3. **Monitoring**: Track costs in production
4. **Async**: Use async for concurrent requests
5. **Error Handling**: Handle specific error types appropriately
6. **Model Selection**: Choose model based on speed/quality/cost tradeoff
7. **Streaming**: Use for large responses to reduce latency
8. **Retry**: Configure appropriate retry attempts for your use case

---

For more examples, see the `examples/` directory.
