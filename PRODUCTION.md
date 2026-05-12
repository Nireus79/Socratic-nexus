# Production Deployment Guide - Socratic Nexus

A production-ready universal LLM client library with 70%+ test coverage and comprehensive error handling.

## Production Readiness Checklist

### Code Quality & Testing
- [x] 70.94% test coverage (62 test files)
- [x] Full async/sync support (no single-threaded limitations)
- [x] Comprehensive error handling (unified APIError)
- [x] Type hints (mypy compatible)
- [x] Multiple provider implementations tested
- [x] Retry logic with exponential backoff
- [x] Token tracking and cost estimation

### Security & Authentication
- [ ] Set secure API key management:
  ```python
  import os
  from socratic_nexus import ClaudeClient
  
  # Use environment variables, not hardcoded
  api_key = os.getenv('ANTHROPIC_API_KEY')
  client = ClaudeClient(api_key=api_key)
  ```
- [ ] Rotate API keys regularly
- [ ] Use separate API keys per environment (dev, staging, prod)
- [ ] Monitor API key usage for suspicious activity

### Configuration Management
```python
# Production configuration example
client = ClaudeClient(
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    max_retries=3,
    timeout=30.0,
    base_url=os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
)

# For cost-sensitive production: use token tracking
response = await client.generate(
    model='claude-3-sonnet',
    messages=[...],
    max_tokens=1000,  # Cap tokens to control costs
)
print(f"Cost: ${response.usage.estimated_cost}")
```

### Deployment Patterns

#### Single Provider Deployment
```python
# Use specific client for maximum efficiency
from socratic_nexus import ClaudeClient

client = ClaudeClient(api_key=os.getenv('ANTHROPIC_API_KEY'))
```

#### Multi-Provider Deployment (Fallback Strategy)
```python
from socratic_nexus import ClaudeClient, OpenAIClient

primary_client = ClaudeClient(api_key=os.getenv('ANTHROPIC_API_KEY'))
fallback_client = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))

try:
    response = await primary_client.generate(...)
except Exception as e:
    logger.warning(f"Primary client failed: {e}, using fallback")
    response = await fallback_client.generate(...)
```

#### Local Development (Ollama)
```python
from socratic_nexus import OllamaClient

# No API key needed for local Ollama
client = OllamaClient(base_url='http://localhost:11434')
response = await client.generate(model='llama2', ...)
```

### Monitoring & Observability

#### Token Usage Tracking
```python
# Track costs across requests
from socratic_nexus import ClaudeClient

client = ClaudeClient(api_key=api_key)
response = await client.generate(...)

total_cost = response.usage.estimated_cost
input_tokens = response.usage.input_tokens
output_tokens = response.usage.output_tokens

logger.info(f"Generation cost: ${total_cost}, tokens: {input_tokens}→{output_tokens}")
```

#### Performance Monitoring
```python
import time

start = time.time()
response = await client.generate(...)
duration = time.time() - start

metrics.record('llm_generation_duration_seconds', duration)
metrics.record('llm_tokens_generated', response.usage.output_tokens)
metrics.record('llm_generation_cost', response.usage.estimated_cost)
```

#### Error Handling
```python
from socratic_nexus.exceptions import APIError

try:
    response = await client.generate(...)
except APIError as e:
    # Log the error with context
    logger.error(
        "LLM generation failed",
        extra={
            'provider': 'anthropic',
            'model': 'claude-3-sonnet',
            'error': str(e),
            'retry_count': e.retry_count if hasattr(e, 'retry_count') else 0,
        }
    )
    # Implement graceful degradation
    response = default_response()
```

### Async Best Practices

```python
import asyncio
from socratic_nexus import ClaudeClient

client = ClaudeClient(api_key=api_key)

# Correct: Use async context
async def batch_generate(prompts):
    tasks = [
        client.generate(model='claude-3-sonnet', messages=[{'role': 'user', 'content': p}])
        for p in prompts
    ]
    return await asyncio.gather(*tasks)

# Run batch
results = asyncio.run(batch_generate(prompts))
```

### Cost Optimization

```python
# 1. Use model pricing to choose cost-effective provider
# Claude 3 Sonnet: $3/1M input, $15/1M output
# GPT-4 Turbo: $10/1M input, $30/1M output
# For cost-sensitive: use Llama or Mistral via Ollama

# 2. Limit token usage
response = await client.generate(
    model='claude-3-sonnet',
    messages=[...],
    max_tokens=500,  # Cap output tokens
)

# 3. Cache responses for identical requests
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_response(prompt: str):
    return asyncio.run(client.generate(
        model='claude-3-sonnet',
        messages=[{'role': 'user', 'content': prompt}]
    ))

# 4. Track costs across organizations
def track_cost(provider: str, cost: float, organization_id: str):
    db.record_cost(
        provider=provider,
        cost=cost,
        organization_id=organization_id,
        timestamp=datetime.now(),
    )
```

### Framework Integration

#### With LangChain
```python
from socratic_nexus.integrations.langchain import socratic_llm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Use Socratic client as LangChain LLM
prompt = PromptTemplate(...)
chain = LLMChain(llm=socratic_llm, prompt=prompt)
```

#### With LangGraph
```python
from socratic_nexus.integrations.langgraph import SocraticNode
from langgraph.graph import StateGraph

# Build multi-step workflows
graph = StateGraph(AgentState)
graph.add_node("generate", SocraticNode(client))
```

### Scaling Considerations

**Concurrency**: The library is fully async-capable
- Use connection pooling for HTTP clients
- Manage rate limits per provider
- Implement queue-based request batching

**Cost Management**:
- Monitor per-provider costs
- Implement spending limits
- Use cheaper models for non-critical tasks
- Cache common responses

**Reliability**:
- Use fallback providers
- Implement retry logic (built-in with backoff)
- Log all API interactions for debugging
- Monitor provider API status

### Examples & Further Learning

See `/examples/` directory for 16 working examples:
- Basic provider usage
- Streaming responses
- Async/concurrent requests
- Error handling
- Cost tracking
- Provider fallback
- Integration with frameworks

