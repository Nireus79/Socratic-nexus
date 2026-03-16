# Socrates Nexus - Troubleshooting

## Connection Issues

### Error: "Connection timeout"

**Symptoms**: Request hangs or times out after 60 seconds

**Causes**:
- API endpoint is slow or unresponsive
- Network issue
- Default timeout too short for your use case

**Solutions**:

1. Increase timeout:
```python
client = LLMClient(
    provider="anthropic",
    model="claude-opus",
    request_timeout=120  # 2 minutes
)
```

2. Check network:
```bash
# Test connectivity to API endpoint
ping api.anthropic.com
```

3. Use smaller prompts initially

### Error: "Connection refused"

**Causes**: 
- API endpoint is down
- Wrong base_url

**Solution**:
```python
# Verify endpoint is correct
client = LLMClient(
    provider="anthropic",
    model="claude-opus",
    base_url="https://api.anthropic.com"  # Correct
)
```

---

## Authentication Issues

### Error: "Invalid API key"

**Symptoms**: Immediate 401 error on any request

**Causes**:
- API key is wrong
- API key expired
- API key doesn't have required permissions

**Solutions**:

1. Verify API key:
```bash
# Check environment variable
echo $ANTHROPIC_API_KEY
```

2. Get new API key from provider console

3. Pass explicitly:
```python
client = LLMClient(
    provider="anthropic",
    model="claude-opus",
    api_key="sk-ant-correct-key-here"
)
```

### Error: "Unauthorized"

**Cause**: API key valid but lacks permissions

**Solution**: Check provider console for required scopes/permissions

---

## Rate Limiting

### Error: "Rate limit exceeded" (HTTP 429)

**Symptoms**: Occasional failures on repeated requests

**Causes**:
- Too many requests in short time
- Shared API account
- Hitting provider's rate limits

**Solutions**:

1. Automatic retries (enabled by default):
```python
client = LLMClient(
    provider="anthropic",
    retry_attempts=5,  # Increase retries
    retry_backoff_factor=2.0
)
```

2. Manual rate limiting:
```python
import time

def safe_chat(message):
    while True:
        try:
            return client.chat(message)
        except RateLimitError as e:
            time.sleep(e.retry_after or 5)
```

3. Batch requests with delays:
```python
import time

for query in queries:
    response = client.chat(query)
    time.sleep(1)  # Rate limiting
```

---

## Model Issues

### Error: "Model not found"

**Causes**:
- Model name is incorrect
- Model doesn't exist
- Model not available in region

**Solutions**:

Check correct model names per provider:

**Anthropic**:
```python
# Correct names
"claude-opus"           # Not "claude-opus-4"
"claude-3-sonnet"       # Not "claude-3-sonnet-20240229"
"claude-3-5-sonnet"
"claude-3-haiku"
```

**OpenAI**:
```python
"gpt-4"
"gpt-4o"
"gpt-3.5-turbo"
```

**Google**:
```python
"gemini-1.5-pro"
"gemini-1.5-flash"
```

---

## Input Issues

### Error: "Context length exceeded"

**Symptoms**: Error when sending long prompts

**Cause**: Input exceeds model's max tokens

**Solutions**:

1. Truncate input:
```python
from socrates_nexus import ContextLengthExceededError

try:
    response = client.chat(long_text)
except ContextLengthExceededError:
    truncated = long_text[:4000]  # Truncate
    response = client.chat(truncated)
```

2. Use model with larger context:
```python
# Claude Opus has 200K context
# GPT-4 Turbo has 128K context
client = LLMClient(provider="anthropic", model="claude-opus")
```

3. Split into chunks:
```python
chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
responses = [client.chat(chunk) for chunk in chunks]
```

---

## Streaming Issues

### Streaming seems slow

**Causes**:
- Model is slow
- Callback is slow
- Network is slow

**Solution**:
```python
import time

def fast_callback(chunk):
    # Fast callback - don't do heavy processing
    print(chunk, end="", flush=True)

response = client.stream("query", on_chunk=fast_callback)
```

### Streaming stops without error

**Cause**: Provider closed connection

**Solution**:
```python
def safe_stream(message):
    try:
        response = client.stream(message, on_chunk=print)
        return response
    except Exception as e:
        logger.error(f"Stream failed: {e}")
        return None
```

---

## Cost Issues

### Unexpected high costs

**Causes**:
- Many requests
- Long responses
- Expensive model

**Solution**:
```python
# Monitor costs
response = client.chat("query")
if response.usage.cost_usd > 0.01:
    logger.warning("Expensive request!")

# Use cheaper models
cheap = LLMClient(provider="openai", model="gpt-3.5-turbo")
expensive = LLMClient(provider="anthropic", model="claude-opus")
```

### Cache not working

**Symptoms**: Repeated queries still cost money

**Solution**:
```python
client = LLMClient(
    provider="anthropic",
    cache_responses=True,  # Must be True
    cache_ttl=3600         # Must be > 0
)

# Same exact query should hit cache
response1 = client.chat("What is Python?")  # API call
response2 = client.chat("What is Python?")  # Cache hit
```

---

## Async Issues

### "Event loop is closed"

**Cause**: Async code running after loop closed

**Solution**:
```python
import asyncio

async def main():
    client = AsyncLLMClient(provider="anthropic")
    response = await client.chat("query")
    return response

result = asyncio.run(main())
```

### "RuntimeError: no running event loop"

**Cause**: Async function called from sync code

**Solution**:
```python
# Use sync client in sync code
client = LLMClient(provider="anthropic")
response = client.chat("query")

# Use async client in async code
async def async_chat():
    client = AsyncLLMClient(provider="anthropic")
    response = await client.chat("query")
    return response
```

---

## Debugging

### Enable verbose logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("socrates_nexus")
logger.setLevel(logging.DEBUG)

# Now all requests are logged
client = LLMClient(provider="anthropic", model="claude-opus")
response = client.chat("test")
```

### Print request details

```python
response = client.chat("query")

print(f"Provider: {response.provider}")
print(f"Model: {response.model}")
print(f"Stop reason: {response.stop_reason}")
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Cost: ${response.usage.cost_usd}")
```

### Check configuration

```python
print(client.config.provider)
print(client.config.model)
print(client.config.retry_attempts)
print(client.config.cache_ttl)
```
