# Socratic Nexus - Architecture Guide

Comprehensive documentation of the internal architecture and design patterns in Socratic Nexus.

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Client Implementations](#client-implementations)
4. [Data Flow](#data-flow)
5. [Design Patterns](#design-patterns)
6. [Extension Points](#extension-points)
7. [Framework Integrations](#framework-integrations-v035)
8. [Performance Considerations](#performance-considerations)

## System Overview

Socratic Nexus provides a unified abstraction layer over multiple LLM providers through a consistent client interface. The system is designed with separation of concerns, allowing each provider implementation to be independent while maintaining API consistency.

### Architecture Diagram

```
┌─────────────────────────────────────┐
│     Application Code                │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│     Client Interface Layer          │
│  ┌──────────────────────────────┐   │
│  │ ClaudeClient                 │   │
│  │ OpenAIClient                 │   │
│  │ GoogleClient                 │   │
│  │ OllamaClient                 │   │
│  └──────────────────────────────┘   │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│  Base Client & Utilities            │
│  ┌──────────────────────────────┐   │
│  │ Caching System               │   │
│  │ Token Tracking               │   │
│  │ Retry Logic                  │   │
│  │ Error Handling               │   │
│  └──────────────────────────────┘   │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  Provider-Specific Implementations                  │
│  ┌──────────────┬──────────────┬──────────────┐     │
│  │ Anthropic    │ OpenAI       │ Google       │     │
│  │ - genai      │ - openai     │ - genai      │     │
│  │ - Client SDK │ - Client SDK │ - Client SDK │     │
│  └──────────────┴──────────────┴──────────────┘     │
│  ┌──────────────┐                                   │
│  │ Ollama       │                                   │
│  │ - HTTP API   │                                   │
│  └──────────────┘                                   │
└──────────────────┬───────────────────────────────────┘
                   │
┌──────────────────▼──────────────────┐
│  External LLM Provider APIs         │
│  ┌──────────────────────────────┐   │
│  │ Anthropic API                │   │
│  │ OpenAI API                   │   │
│  │ Google Gemini API            │   │
│  │ Ollama HTTP API              │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Core Components

### 1. Client Interface Layer

All clients inherit from a common base with these key methods:

```python
class BaseClient:
    # Core generation methods
    def generate_response(
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str

    def generate_code(
        specification: str,
        language: str = "python"
    ) -> str

    def extract_insights(
        text: str,
        project: Optional[ProjectContext] = None
    ) -> str

    # Async variants
    async def generate_response_async(...) -> str
    async def generate_code_async(...) -> str
    async def extract_insights_async(...) -> str

    # Configuration and utilities
    def test_connection() -> bool
    def get_auth_credential(...) -> Optional[str]
```

### 2. Client Implementations

Each client implements provider-specific logic while maintaining the common interface:

#### ClaudeClient
- **File**: `src/socratic_nexus/clients/claude_client.py`
- **Dependencies**: `anthropic>=0.40.0`
- **Features**:
  - 9 specialized generation methods (business plans, curriculum, documentation, etc.)
  - Full async/sync support
  - Token cost tracking with accurate pricing
  - Intelligent caching system
  - Graceful API key validation

**Specialized Methods**:
- `generate_business_plan()` - Create business strategies
- `generate_curriculum()` - Design learning paths
- `generate_documentation()` - Generate technical docs
- `generate_research_protocol()` - Research methodology
- `generate_marketing_plan()` - Marketing strategies
- `generate_creative_brief()` - Creative direction
- `generate_artifact()` - Multi-media artifacts
- `generate_conflict_resolution_suggestions()` - Conflict analysis
- `generate_suggestions()` - General suggestions

#### OpenAIClient
- **File**: `src/socratic_nexus/clients/openai_client.py`
- **Dependencies**: `openai>=1.0.0`
- **Features**:
  - GPT-4, GPT-4 Turbo, and GPT-3.5 support
  - Token cost calculation based on actual pricing
  - Configurable model selection
  - Full async support
  - Streaming support ready

#### GoogleClient
- **File**: `src/socratic_nexus/clients/google_client.py`
- **Dependencies**: `google-generativeai>=0.1.0rc1`
- **Features**:
  - Gemini API support
  - Multi-modal capabilities
  - Safety settings configuration
  - Model variant selection (pro, pro-vision)
  - Full async support with thread pooling

#### OllamaClient
- **File**: `src/socratic_nexus/clients/ollama_client.py`
- **Dependencies**: `requests`
- **Features**:
  - Local LLM support (no API keys required)
  - Custom model selection
  - HTTP-based communication
  - Development-friendly (runs locally)
  - Full async support with thread pooling

### 3. Shared Infrastructure

#### Caching System
- Implements in-memory LRU caching
- Configurable cache size and TTL
- Separates response cache from insight cache
- Cache key generation from prompts
- Optional Redis integration ready

#### Token Tracking
- Monitors input/output token usage
- Calculates costs based on provider pricing
- Events emitted for tracking systems
- Timestamp recording for analytics

#### Error Handling
- Provider-specific error mapping
- Graceful fallback for unavailable APIs
- Detailed error messages
- Retry logic with exponential backoff

#### Authentication
- Multiple auth methods per provider (API key, subscription, OAuth)
- Secure credential handling
- Placeholder key detection
- Database-backed credential storage (optional)

## Data Flow

### Request Processing Pipeline

```
1. Client Method Call
   │
2. Input Validation
   │
3. Cache Lookup
   ├─ Hit: Return cached response
   └─ Miss: Continue
   │
4. Provider Initialization
   │
5. Request Formatting
   │
6. API Call with Retry Logic
   │
7. Response Parsing
   │
8. Token Usage Tracking
   │
9. Cache Storage
   │
10. Return to Caller
```

### Async Execution Flow

For async clients, the flow is identical but non-blocking:

```python
# Sync - Blocking
response = client.generate_response(prompt)

# Async - Non-blocking
response = await client.generate_response_async(prompt)
# or
task = asyncio.create_task(client.generate_response_async(prompt))
# ... do other work ...
response = await task
```

For Ollama and Google clients, async is achieved through `asyncio.to_thread()` for thread safety.

## Design Patterns

### 1. Adapter Pattern

Each provider is an adapter that:
- Converts unified interface to provider-specific API
- Transforms provider responses to common format
- Handles provider-specific error codes
- Manages provider-specific authentication

```python
# Unified interface
response = client.generate_response(prompt)

# Internally adapts to provider
# - Claude: calls anthropic.Anthropic.messages.create()
# - OpenAI: calls openai.OpenAI().chat.completions.create()
# - Google: calls genai.GenerativeModel.generate_content()
# - Ollama: calls /api/generate HTTP endpoint
```

### 2. Factory Pattern

Clients are instantiated with sensible defaults but accept configuration:

```python
# Default configuration
client = ClaudeClient(api_key="...")

# Custom configuration
client = ClaudeClient(
    api_key="...",
    model="claude-3-5-haiku-20241022",
    max_tokens=2048,
    temperature=0.5,
    timeout=60
)
```

### 3. Strategy Pattern

Different strategies per provider:

- **Model Selection**: Each provider has different available models
- **Async Implementation**: Google and Ollama use thread pooling; others use native async
- **Cost Calculation**: Each provider has different pricing models
- **Error Handling**: Provider-specific error codes and recovery strategies

### 4. Decorator Pattern

Methods are decorated for:

- **Caching**: Intelligent cache management
- **Retry Logic**: Exponential backoff on failures
- **Monitoring**: Token usage tracking
- **Logging**: Debug and error logging

### 5. Template Method Pattern

Base client defines algorithm structure:

```python
def generate_response(self, prompt, **kwargs):
    # Template method outline
    self._validate_input(prompt)        # Step 1
    cached = self._get_from_cache(key)  # Step 2
    if cached:
        return cached
    response = self._call_api(prompt)   # Step 3 (overridden by subclass)
    self._track_tokens(response)        # Step 4
    self._store_in_cache(key, response) # Step 5
    return response
```

Each client overrides `_call_api()` with provider-specific implementation.

## Extension Points

### Adding a New Provider

To add a new LLM provider (e.g., Cohere):

1. **Create new client class**:
   ```python
   # src/socratic_nexus/clients/cohere_client.py
   from socratic_nexus.clients.base import BaseClient

   class CohereClient(BaseClient):
       def __init__(self, api_key, **kwargs):
           super().__init__(**kwargs)
           self.api_key = api_key
           self.client = cohere.Client(api_key)

       def _call_api(self, prompt, **kwargs):
           # Cohere-specific implementation
           response = self.client.generate(prompt)
           return response.generations[0].text
   ```

2. **Add to package exports**:
   ```python
   # src/socratic_nexus/clients/__init__.py
   from .cohere_client import CohereClient
   ```

3. **Add tests**:
   ```python
   # tests/test_cohere_client.py
   class TestCohereClient:
       def test_generation(self):
           ...
   ```

### Custom Client Behavior

Extend any client with custom behavior:

```python
class CustomClaudeClient(ClaudeClient):
    def _track_tokens(self, response):
        # Custom token tracking
        super()._track_tokens(response)
        # Send to custom analytics
        self.analytics.record(response.usage)
```

## Framework Integrations (v0.3.5+)

Socratic Nexus includes built-in integrations with popular AI frameworks:

### LangChain Integration

**File**: `src/socratic_nexus/integrations/langchain.py`

Provides `SocratesNexusLLM` wrapper for use in LangChain applications:

```python
from socratic_nexus.clients import ClaudeClient
from socratic_nexus.integrations.langchain import SocratesNexusLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

client = ClaudeClient(api_key="...")
llm = SocratesNexusLLM(client=client)

prompt = PromptTemplate(template="Explain {topic}", input_variables=["topic"])
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="Machine Learning")
```

**Features**:
- Full LangChain `LLM` base class compatibility
- Async support with `_generate_async()`
- Token counting heuristics
- Multi-provider support (Claude, OpenAI, Google, Ollama)
- Temperature and max_tokens configuration

### LangGraph Integration

**File**: `src/socratic_nexus/integrations/langgraph.py`

Provides factory functions for creating LangGraph workflow nodes:

```python
from socratic_nexus.clients import ClaudeClient
from socratic_nexus.integrations.langgraph import (
    create_nexus_node,
    create_nexus_agent,
    create_routing_node
)

client = ClaudeClient(api_key="...")

# Create a workflow node
node = create_nexus_node(
    client=client,
    node_name="analyze",
    prompt_key="content",
    output_key="analysis"
)

# Create a stateful agent
agent = create_nexus_agent(
    client=client,
    system_prompt="You are helpful."
)

# Create a routing node
router = create_routing_node(
    client=client,
    routes={"summarize": "...", "analyze": "..."}
)
```

**Features**:
- State dict-based interface for LangGraph compatibility
- Factory functions for common patterns
- Error handling with fallback responses
- System prompt support
- Custom state key mapping

### Openclaw Integration

**File**: `src/socratic_nexus/integrations/openclaw.py`

Provides skill wrappers for Openclaw framework:

```python
from socratic_nexus.clients import ClaudeClient
from socratic_nexus.integrations.openclaw import (
    NexusLLMSkill,
    NexusAnalysisSkill,
    NexusCodeGenSkill,
    NexusDocumentationSkill
)

client = ClaudeClient(api_key="...")

# Basic LLM skill
skill = NexusLLMSkill(client=client, name="analyzer")
response = skill.query("Analyze this text...")

# Specialized skills
analysis = NexusAnalysisSkill(client=client)
code = NexusCodeGenSkill(client=client)
docs = NexusDocumentationSkill(client=client)

# Structured processing
result = skill.process({"prompt": "...", "context": "..."})
```

**Features**:
- Skill base class implementing Openclaw interface
- Specialized skill subclasses for common tasks
- `query()` and `query_async()` methods
- `process()` method for structured workflows
- Skill metadata and introspection via `get_info()`
- System prompt support

### Integration Architecture

All integrations follow this pattern:

```
Application Framework (LangChain/LangGraph/Openclaw)
        ↓
    Integration Module
        ↓
    Socratic Nexus Client (ClaudeClient, OpenAIClient, etc.)
        ↓
    Provider APIs (Anthropic, OpenAI, Google, etc.)
```

**Benefits**:
- No lock-in to a single framework
- Consistent client interface across integrations
- Async support throughout the stack
- Full token tracking and cost estimation
- Error handling at each layer

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Cache lookup | O(1) | Hash-based cache |
| API call | O(n) | n = response size |
| Token counting | O(n) | n = token count |

### Space Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Response cache | O(k) | k = number of cached responses |
| Client instances | O(p) | p = number of providers |

### Memory Usage

- Single client instance: ~5-10 MB
- With full cache (1000 responses): ~50-100 MB
- Async tasks overhead: Minimal (< 1 MB per task)

### Optimization Strategies

1. **Caching**: Reuse responses for identical prompts
2. **Async**: Handle multiple requests concurrently
3. **Connection Pooling**: Reuse HTTP connections
4. **Lazy Loading**: Initialize providers on-demand
5. **Token Optimization**: Use smaller models when appropriate

## Testing Architecture

### Test Organization

```
tests/
├── test_all_llm_clients.py          # Basic client tests
├── test_clients_error_handling.py   # Error scenarios
├── test_clients_methods.py          # Method implementations
├── test_coverage_gaps.py            # Coverage-focused tests
└── [provider-specific tests]
```

### Test Coverage

- **Target**: 70%+ coverage
- **Current**: 70.94% coverage
- **Focus areas**:
  - Client initialization
  - API method signatures
  - Error handling
  - Cache functionality
  - Token tracking

### Running Tests

```bash
# All tests
pytest tests/

# Specific test class
pytest tests/test_all_llm_clients.py::TestGoogleClientBasic

# With coverage
pytest --cov=socratic_nexus tests/

# By marker
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
```

## Security Architecture

### API Key Management

- Never logged or stored in code
- Loaded from environment variables
- Validated before use
- Placeholder keys detected and rejected

### Data Handling

- No request/response caching of sensitive data
- Input validation before sending
- HTTPS-only communication
- No sensitive data in debug logs

### Error Messages

- Provider errors don't expose API internals
- Generic error messages to users
- Detailed logs only in debug mode

## Deployment Considerations

### Single-Machine Deployment

```python
# Simple usage
client = ClaudeClient(api_key="...")
response = client.generate_response("prompt")
```

### Multi-Process Deployment

```python
# Each process gets its own client instance
from multiprocessing import Pool

def process_prompt(prompt):
    client = ClaudeClient()  # Initialized per process
    return client.generate_response(prompt)

with Pool(4) as pool:
    results = pool.map(process_prompt, prompts)
```

### Async/Concurrent Usage

```python
import asyncio

async def main():
    client = ClaudeClient()

    # Concurrent requests
    responses = await asyncio.gather(
        client.generate_response_async("prompt 1"),
        client.generate_response_async("prompt 2"),
        client.generate_response_async("prompt 3")
    )

asyncio.run(main())
```

## Future Roadmap

- [ ] Streaming response support
- [ ] Function calling for all providers
- [ ] Load balancing across providers
- [ ] Advanced caching strategies (distributed)
- [ ] Rate limiting and quota management
- [ ] Provider switching with fallback
- [ ] Cost optimization routing

---

For API documentation, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
