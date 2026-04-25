# Framework Integrations Guide

Comprehensive guide for integrating Socratic Nexus clients with LangChain, LangGraph, and Openclaw frameworks.

## Installation

```bash
# Base package
pip install socratic-nexus

# With LangChain support
pip install socratic-nexus[langchain]

# All integrations
pip install socratic-nexus[full]
```

---

## LangChain Integration

### Overview

Use Socratic Nexus clients with LangChain chains, agents, and other components. The `SocratesNexusLLM` wrapper makes any Socratic Nexus client compatible with LangChain.

**Benefits for Openclaw users**:
- Multi-provider LLM support (not locked to one provider)
- Easy provider switching
- Automatic retry logic
- Token tracking and cost monitoring
- Streaming support

### Basic Usage

```python
from socrates_nexus.integrations.openclaw import NexusLLMSkill

# Create skill with default provider (Anthropic Claude)
skill = NexusLLMSkill()

# Query
response = skill.query("What is machine learning?")
print(response.content)
print(f"Cost: ${response.usage.cost_usd}")
```

### Custom Configuration

```python
skill = NexusLLMSkill(
    provider="anthropic",
    model="claude-opus",
    temperature=0.7,
    max_tokens=1024,
    retry_attempts=3,
    cache_responses=True,
    cache_ttl=300,  # 5 minutes
)
```

### Provider Switching

```python
# Query with Claude
response1 = skill.query("Explain recursion")
print(f"Claude: {response1.content}")

# Switch to OpenAI
skill.switch_provider("openai", "gpt-4")
response2 = skill.query("Explain recursion")
print(f"GPT-4: {response2.content}")

# Switch to Google
skill.switch_provider("google", "gemini-pro")
response3 = skill.query("Explain recursion")
print(f"Gemini: {response3.content}")
```

### Streaming Support

```python
def on_chunk(chunk):
    print(chunk, end="", flush=True)

response = skill.stream("Write a haiku about AI", on_chunk=on_chunk)
print(f"\n\nCost: ${response.usage.cost_usd}")
```

### Usage Tracking

```python
# Track per-request usage
response = skill.query("Query")
print(f"Input: {response.usage.input_tokens}")
print(f"Output: {response.usage.output_tokens}")
print(f"Cost: ${response.usage.cost_usd}")

# Get cumulative stats
stats = skill.get_usage_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Total cost: ${stats['total_cost_usd']}")
print(f"By provider: {stats['by_provider']}")
```

### Custom Callbacks

```python
def track_expensive_requests(usage):
    if usage.cost_usd > 0.01:
        print(f"Expensive request: ${usage.cost_usd:.6f}")

skill.add_usage_callback(track_expensive_requests)

# Now all requests will be tracked
skill.query("Query")
skill.query("Another query")
```

### Complete Example

```python
from socrates_nexus.integrations.openclaw import NexusLLMSkill

# Initialize skill
skill = NexusLLMSkill(
    provider="anthropic",
    model="claude-opus",
    cache_responses=True,
)

# Add usage tracking
def log_usage(usage):
    print(f"[Usage] {usage.total_tokens} tokens, ${usage.cost_usd:.6f}")

skill.add_usage_callback(log_usage)

# Process multiple queries
queries = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?",
]

for q in queries:
    response = skill.query(q)
    print(f"Q: {q}\nA: {response.content[:100]}...\n")

# Check stats
stats = skill.get_usage_stats()
print(f"Total: {stats['total_requests']} requests, ${stats['total_cost_usd']:.2f}")
```

---

## LangChain Integration

### Overview

The LangChain integration provides `SocratesNexusLLM`, a LangChain-compatible LLM that brings multi-provider support to your chains.

**Benefits for LangChain users**:
- Drop-in replacement for OpenAI LLM
- Multi-provider support in all chains
- Automatic retry logic
- Token tracking
- Easy provider switching in chains

### Basic Usage

```python
from socrates_nexus.integrations.langchain import SocratesNexusLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Create LLM
llm = SocratesNexusLLM(provider="anthropic", model="claude-opus")

# Create chain
prompt = PromptTemplate(
    template="Explain {topic} in one sentence",
    input_variables=["topic"]
)
chain = LLMChain(llm=llm, prompt=prompt)

# Run chain
result = chain.run(topic="machine learning")
print(result)
```

### Multi-Provider Chains

```python
from socrates_nexus.integrations.langchain import SocratesNexusLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    template="Q: {question}\nA:",
    input_variables=["question"]
)

providers = [
    ("anthropic", "claude-opus"),
    ("openai", "gpt-4"),
    ("google", "gemini-pro"),
]

question = "What is AI?"

for provider, model in providers:
    llm = SocratesNexusLLM(provider=provider, model=model)
    chain = LLMChain(llm=llm, prompt=template)
    try:
        result = chain.run(question=question)
        print(f"{provider}: {result}")
    except Exception as e:
        print(f"{provider}: Failed ({str(e)[:50]})")
```

### Provider Fallback Pattern

```python
from socrates_nexus.integrations.langchain import SocratesNexusLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    template="{prompt}",
    input_variables=["prompt"]
)

providers = [
    ("anthropic", "claude-opus"),
    ("openai", "gpt-4"),
    ("google", "gemini-pro"),
]

def run_with_fallback(prompt, providers):
    for provider, model in providers:
        try:
            llm = SocratesNexusLLM(provider=provider, model=model)
            chain = LLMChain(llm=llm, prompt=template)
            result = chain.run(prompt=prompt)
            print(f"Success with {provider}")
            return result
        except Exception as e:
            print(f"{provider} failed, trying next...")
            continue
    raise Exception("All providers failed")

result = run_with_fallback("Write a poem about AI", providers)
print(result)
```

### Custom Configuration

```python
llm = SocratesNexusLLM(
    provider="anthropic",
    model="claude-opus",
    temperature=0.9,           # More creative
    max_tokens=2048,
    retry_attempts=5,
    cache_responses=True,
    cache_ttl=600,            # 10 minutes
    api_key="sk-ant-...",     # Optional, uses env var if not provided
)
```

### Q&A Chain Example

```python
from socrates_nexus.integrations.langchain import SocratesNexusLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

qa_template = """Answer the following question concisely:

Question: {question}

Answer:"""

llm = SocratesNexusLLM(provider="anthropic", model="claude-opus")
prompt = PromptTemplate(template=qa_template, input_variables=["question"])
chain = LLMChain(llm=llm, prompt=prompt)

questions = [
    "What is Python?",
    "How do you reverse a string?",
    "What are decorators?",
]

for q in questions:
    answer = chain.run(question=q)
    print(f"Q: {q}\nA: {answer}\n")
```

### Complete Example

```python
from socrates_nexus.integrations.langchain import SocratesNexusLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Create prompt
template = PromptTemplate(
    template="Complete this story: {beginning}",
    input_variables=["beginning"]
)

# Create LLM with custom config
llm = SocratesNexusLLM(
    provider="anthropic",
    model="claude-opus",
    temperature=0.8,
    max_tokens=500,
    cache_responses=True,
)

# Create chain
chain = LLMChain(llm=llm, prompt=template)

# Run chain
story_beginning = "Once upon a time, a robot discovered"
result = chain.run(beginning=story_beginning)
print(result)

# Easy provider switching
print("\n--- Now with GPT-4 ---\n")
llm2 = SocratesNexusLLM(provider="openai", model="gpt-4")
chain2 = LLMChain(llm=llm2, prompt=template)
result2 = chain2.run(beginning=story_beginning)
print(result2)
```

---

## Comparison: Integration Approaches

### Openclaw Skill (Recommended for Openclaw)
**Pros**:
- Native Openclaw integration
- Simple, Openclaw-native API
- Easy provider switching
- Great for multi-agent systems

**Use When**:
- Building within Openclaw framework
- Need Openclaw-specific features

### LangChain LLM (Recommended for LangChain)
**Pros**:
- Drop-in replacement for OpenAI LLM
- Works in all LangChain chains
- Great for complex chains and agents
- Full LangChain ecosystem access

**Use When**:
- Using LangChain chains
- Building complex AI workflows
- Need agent support

### Standalone Client (Recommended for both)
**Pros**:
- No framework dependencies
- Maximum flexibility
- Use with any framework
- Easy to integrate

**Use When**:
- Building custom solutions
- Using multiple frameworks
- Maximum control needed

---

## Best Practices

### 1. Use Optional Dependencies

Always use extras when installing:
```bash
# For Openclaw users
pip install socrates-nexus[openclaw]

# For LangChain users
pip install socrates-nexus[langchain]

# Users who need both
pip install socrates-nexus[integrations]
```

### 2. Handle Provider Errors

```python
from socrates_nexus import RateLimitError, AuthenticationError

try:
    response = skill.query("Query")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except AuthenticationError:
    print("Invalid API key")
except Exception as e:
    print(f"Error: {e}")
```

### 3. Monitor Token Usage

```python
# Add callback for tracking
def track_usage(usage):
    print(f"Used {usage.total_tokens} tokens (${usage.cost_usd:.6f})")

skill.add_usage_callback(track_usage)

# Or check stats periodically
stats = skill.get_usage_stats()
print(f"Budget used: ${stats['total_cost_usd']:.2f}")
```

### 4. Cache Responses

```python
# Enable caching to save costs
skill = NexusLLMSkill(
    cache_responses=True,
    cache_ttl=600,  # Cache for 10 minutes
)

# Identical queries reuse cache
response1 = skill.query("What is Python?")  # API call
response2 = skill.query("What is Python?")  # From cache
```

### 5. Implement Fallback Logic

```python
def safe_query(question, providers):
    for provider, model in providers:
        try:
            skill = NexusLLMSkill(provider=provider, model=model)
            return skill.query(question)
        except Exception:
            continue
    raise Exception("All providers failed")

# Try multiple providers in order
providers = [
    ("anthropic", "claude-opus"),
    ("openai", "gpt-4"),
    ("google", "gemini-pro"),
]

result = safe_query("Query", providers)
```

---

## Troubleshooting

### Import Error: "No module named 'langchain'"
Install with LangChain support:
```bash
pip install socrates-nexus[langchain]
```

### AuthenticationError: "API key required"
Provide API key when initializing:
```python
skill = NexusLLMSkill(api_key="sk-ant-...")
# Or set environment variable
# export ANTHROPIC_API_KEY="sk-ant-..."
```

### RateLimitError
The library automatically retries with exponential backoff. To customize:
```python
skill = NexusLLMSkill(
    retry_attempts=5,  # More retries
)
```

---

## Examples

See `examples/` directory for complete examples:
- `10_openclaw_integration.py` - Openclaw skill usage
- `11_langchain_integration.py` - LangChain integration examples

---

## Support

For issues or questions:
- [GitHub Issues](https://github.com/Nireus79/socrates-nexus/issues)
- [GitHub Discussions](https://github.com/Nireus79/socrates-nexus/discussions)
