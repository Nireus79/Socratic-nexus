# Socrates Nexus - Guides by Role

## For Developers (Extending Socrates Nexus)

### What You Need to Know

You're building features on top of Socrates Nexus or integrating it into a larger system.

### Key Concepts

1. **Provider Pattern**: All LLM calls go through a provider. New providers implement BaseProvider.

2. **Sync vs Async**: 
   - Use LLMClient for sync code
   - Use AsyncLLMClient for concurrent/async code

3. **Configuration Flow**:
   ```python
   config = LLMConfig(provider="anthropic", model="claude-opus", api_key="...")
   client = LLMClient(config=config)
   ```

4. **Error Handling**: Catch specific exceptions for recovery logic:
   ```python
   try:
       response = client.chat("query")
   except RateLimitError as e:
       # Implement backoff
       wait(e.retry_after)
   except AuthenticationError:
       # Handle auth failure
       refresh_api_key()
   ```

### Adding a New Provider

1. Implement BaseProvider interface
2. Handle provider-specific API formats
3. Implement `extract_usage()` to calculate tokens
4. Implement `calculate_cost()` with provider pricing
5. Add to provider factory

### Custom Integrations

Extend LLMClient for custom behavior:
```python
class CustomClient(LLMClient):
    def _make_request(self, messages, **kwargs):
        # Custom request logic
        return super()._make_request(messages, **kwargs)
```

### Testing

Use mock providers in tests:
```python
from unittest.mock import Mock, patch

mock_response = Response(content="...", usage=TokenUsage(...))
with patch.object(client, 'chat', return_value=mock_response):
    result = client.chat("test")
```

---

## For Data Scientists (Using for ML/Analysis)

### What You Care About

You're using Socrates Nexus for data analysis, experiments, or ML workflows.

### Getting Started

1. **Install**: `pip install socrates-nexus[anthropic]`
2. **Initialize**:
   ```python
   from socrates_nexus import LLMClient
   
   client = LLMClient(provider="anthropic", model="claude-opus")
   ```
3. **Use**: `response = client.chat("Your query")`

### Key Features for Your Workflow

1. **Token Tracking**: Monitor API costs
   ```python
   response = client.chat("query")
   print(f"Cost: ${response.usage.cost_usd}")
   
   stats = client.get_usage_stats()
   print(f"Total spent: ${stats.total_cost_usd}")
   ```

2. **Streaming for Long Outputs**: Get results as they arrive
   ```python
   def print_chunk(chunk):
       print(chunk, end="", flush=True)
   
   response = client.stream("Long query", on_chunk=print_chunk)
   ```

3. **Provider Switching**: Compare outputs from different models
   ```python
   for provider in ["anthropic", "openai", "google"]:
       client = LLMClient(provider=provider, model="...")
       response = client.chat("test query")
       print(f"{provider}: {response.content}")
   ```

4. **Async for Batch Processing**:
   ```python
   import asyncio
   from socrates_nexus import AsyncLLMClient
   
   async def process_batch(queries):
       client = AsyncLLMClient(provider="anthropic")
       results = await asyncio.gather(
           *[client.chat(q) for q in queries]
       )
       return results
   ```

### Common Patterns

**Cost-Optimized Experiments**:
```python
# Use cheaper model for initial experiments
cheap_client = LLMClient(provider="openai", model="gpt-3.5-turbo")
responses = [cheap_client.chat(q) for q in test_queries]

# Use better model for final results
expensive_client = LLMClient(provider="anthropic", model="claude-opus")
final_results = [expensive_client.chat(q) for q in final_queries]
```

**Provider Fallback**:
```python
def safe_query(query):
    for config in provider_configs:
        try:
            client = LLMClient(**config)
            return client.chat(query)
        except Exception:
            continue
    raise Exception("All providers failed")
```

---

## For DevOps (Deploying/Operating)

### What You Need to Manage

You're deploying Socrates Nexus in production and ensuring reliability.

### Configuration Management

1. **Environment Variables**:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   export OPENAI_API_KEY="sk-..."
   export GOOGLE_API_KEY="..."
   ```

2. **Configuration Files**:
   ```yaml
   llm:
     provider: anthropic
     model: claude-opus
     retry_attempts: 3
     request_timeout: 60
     cache_ttl: 300
   ```

### Monitoring & Observability

1. **Usage Tracking**:
   ```python
   def log_usage(usage):
       logger.info(f"tokens={usage.total_tokens} cost=${usage.cost_usd}")
   
   client.add_usage_callback(log_usage)
   ```

2. **Error Rates**:
   - Track RateLimitError frequency (adjust retry config)
   - Monitor AuthenticationError (rotate API keys)
   - Log TimeoutError (adjust timeout setting)

3. **Cost Monitoring**:
   ```python
   stats = client.get_usage_stats()
   if stats.total_cost_usd > daily_budget:
       alert("Daily budget exceeded")
   ```

### Deployment Patterns

1. **Provider Failover**:
   ```python
   # Primary: Claude (best quality)
   # Fallback: GPT-4 (reliable)
   # Fallback: Gemini (cost-effective)
   ```

2. **Rate Limiting**:
   - Socrates Nexus retries automatically (exponential backoff)
   - Configure retry_attempts based on SLA
   - Monitor retry rates

3. **Caching**:
   - Enable for development/testing (saves costs)
   - Disable for production if responses must be fresh
   - Adjust cache_ttl based on content freshness requirements

### Scaling Considerations

1. **Concurrency**:
   ```python
   # Use AsyncLLMClient for high concurrency
   # Default async pool: 10 concurrent requests
   ```

2. **Cost Management**:
   - Monitor cost_usd per request
   - Set up alerts for anomalies
   - Review usage patterns regularly

3. **Reliability**:
   - Retry configuration: 3 attempts, 2.0x backoff
   - Request timeout: 60 seconds (adjust based on model)
   - Provider fallback: Always have 2+ providers configured

---

## For Business Users (What Value They Get)

### What You Benefit From

Faster, more reliable AI applications built on Socrates Nexus.

### Key Benefits

1. **Lower Costs**:
   - Automatic caching reduces API calls
   - Provider switching finds cheapest option
   - Token tracking shows exact spending

2. **Better Reliability**:
   - Automatic retries handle temporary failures
   - Provider fallback if main service is down
   - No interruptions to your service

3. **Faster Development**:
   - Same code works with any LLM provider
   - Easy to switch models without code changes
   - Streaming for better user experience

4. **Detailed Insights**:
   - Know exactly what you're spending on AI
   - Track performance metrics
   - Optimize based on real data

### Real-World Example

**Scenario**: Building a customer support chatbot

**Without Socrates Nexus**:
- Single provider (Claude)
- Manual caching implementation
- Manual retry logic needed
- No cost tracking → billing surprises

**With Socrates Nexus**:
```python
from socrates_nexus import AsyncLLMClient

# Primary: Claude (best for support)
# Fallback: GPT-4 (if Claude is down)
client = AsyncLLMClient(provider="anthropic", model="claude-opus")

# Automatic retries on rate limits
# Automatic cost tracking
# Streaming responses for better UX
response = await client.stream(customer_query, on_chunk=send_to_user)

# Know exact cost per interaction
print(f"This support reply cost: ${response.usage.cost_usd}")
```

**Results**:
- ✅ 30% cost savings (caching + optimization)
- ✅ 99.9% uptime (automatic fallbacks)
- ✅ Better user experience (streaming)
- ✅ Full cost visibility
