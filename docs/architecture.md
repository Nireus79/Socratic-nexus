# Socrates Nexus - Architecture & System Design

## System Overview

Socrates Nexus is a universal LLM client providing a unified interface across multiple LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini, Ollama). Architecture emphasizes production-readiness with automatic retries, token tracking, streaming, and multi-provider fallback.

## Core Components

### 1. LLMClient (Synchronous)
Main synchronous interface implementing retry logic, token tracking, and caching.

**Key Methods**:
- `chat(message, **kwargs)` - Single turn conversation
- `stream(message, on_chunk=None)` - Streaming response  
- `get_usage_stats()` - Cumulative token usage tracking
- `add_usage_callback()` - Register tracking callbacks

### 2. AsyncLLMClient (Asynchronous)
Non-blocking async version for concurrent requests.

**Key Methods**:
- `async chat(message, **kwargs)`
- `async stream(message, on_chunk=None)`
- Same configuration and tracking as sync client

### 3. Provider System
Extensible provider pattern with concrete implementations.

**BaseProvider Interface**:
- `chat()` - Send message to LLM
- `stream()` - Stream response chunks
- `extract_usage()` - Extract token counts
- `calculate_cost()` - Calculate request cost

**Providers**:
- AnthropicProvider: Claude 3 variants
- OpenAIProvider: GPT-4, GPT-3.5-turbo
- GoogleProvider: Gemini 1.5 Pro/Flash
- OllamaProvider: Local LLMs

### 4. Retry System
Automatic retry with exponential backoff for transient failures.

**Features**:
- Configurable retry attempts (default: 3)
- Exponential backoff with jitter
- Retryable: 429, timeouts, 5xx
- Non-retryable: 401, 400, 404

**Backoff Schedule** (2.0x factor):
- Attempt 1: Immediate
- Attempt 2: 1s wait
- Attempt 3: 2s wait
- Attempt 4: 4s wait

### 5. Token Tracking & Caching
Track costs and cache responses to reduce API calls.

**TokenUsage**:
- input_tokens, output_tokens, total_tokens
- cost_usd (calculated per provider pricing)

**Caching**:
- TTL-based (default: 300s)
- Cache key from message hash + model
- In-memory storage

### 6. Streaming System
Real-time response streaming with chunk callbacks.

**StreamHandler**:
- Process provider streaming responses
- Accumulate chunks while calling callback
- Track partial token counts
- Return final usage info

## Data Models

### LLMConfig
```python
@dataclass
class LLMConfig:
    provider: str                    # Provider name
    model: str                       # Model identifier
    api_key: Optional[str]           # API key/env var
    base_url: Optional[str]          # Custom endpoint
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    request_timeout: int = 60
    cache_responses: bool = True
    cache_ttl: int = 300
```

### Response
```python
@dataclass
class Response:
    content: str
    stop_reason: str
    usage: TokenUsage
    model: str
    provider: str
    timestamp: datetime
```

## Error Hierarchy

- NexusError (base)
  - RateLimitError (429)
  - AuthenticationError (401)
  - InvalidAPIKeyError
  - TimeoutError
  - ContextLengthExceededError
  - ModelNotFoundError
  - ProviderError
  - ConfigurationError

## Request Flow

1. Client receives message
2. Check cache (if enabled)
3. Get provider instance
4. Retry loop (0 to retry_attempts):
   - Call provider.chat() with timeout
   - Extract tokens and calculate cost
   - Update cache and invoke callbacks
   - Return Response on success
   - On retryable error: backoff and retry
   - On non-retryable error: raise exception immediately

## Multi-Provider Fallback

```
Primary (e.g., Claude)
├─ Success → Return
└─ Error → Retry with backoff
   └─ Exhausted → Fallback (e.g., GPT-4)
      ├─ Success → Return
      └─ Error → Fallback (e.g., Gemini)
```

## Integration Points

- **Openclaw**: NexusLLMSkill wrapper
- **LangChain**: SocratesNexusLLM base class

## Design Patterns

1. **Provider Pattern** - Extensible provider system
2. **Decorator Pattern** - Retry/streaming/tracking wrappers
3. **Configuration Pattern** - Centralized LLMConfig
4. **Strategy Pattern** - Retry/caching/fallback strategies

## Performance

- Response time: <100ms (cache) to seconds (API)
- Token tracking: O(1)
- Retry overhead: exponential backoff (1s, 2s, 4s)
- Cache hit: ~100x faster than API

## Extensibility

Add providers by implementing BaseProvider:
- `chat(messages, **kwargs)`
- `stream(messages, **kwargs)`
- `extract_usage(response)`
- `calculate_cost(input_tokens, output_tokens)`

Custom integrations via:
- Subclassing LLMClient
- Overriding _make_request()
- Using callbacks
