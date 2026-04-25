# Socratic Nexus Examples

Comprehensive examples demonstrating how to use Socratic Nexus with different LLM providers.

## Quick Start

### 1. Install Socratic Nexus

```bash
pip install socratic-nexus
```

### 2. Set Your API Key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza-..."
```

### 3. Run an Example

```bash
python examples/01_anthropic_basic.py
```

## Examples Overview

### Core Examples

#### 1. **01_anthropic_basic.py** - Claude Basics
- Initialize Claude client
- Generate simple responses
- Basic API usage

**Run:**
```bash
python examples/01_anthropic_basic.py
```

#### 2. **02_openai_gpt4.py** - OpenAI GPT-4
- Initialize OpenAI client
- Generate responses with GPT-4
- Model selection

**Run:**
```bash
python examples/02_openai_gpt4.py
```

#### 3. **03_google_gemini.py** - Google Gemini
- Initialize Google client
- Use Gemini models
- Multi-modal capabilities

**Run:**
```bash
python examples/03_google_gemini.py
```

#### 4. **04_ollama_local.py** - Local Ollama Models
- Run models locally without API keys
- No cost, fully private
- Great for development

**Prerequisites:**
```bash
# Install Ollama from https://ollama.ai
ollama pull llama2
ollama serve
```

**Run:**
```bash
python examples/04_ollama_local.py
```

### Advanced Examples

#### 5. **05_streaming.py** - Streaming Responses
- Stream responses in real-time
- Handle chunks as they arrive
- Progress callbacks

#### 6. **06_async_calls.py** - Asynchronous Usage
- Async/await patterns
- Concurrent requests
- Parallel provider calls

**Run:**
```bash
python examples/06_async_calls.py
```

#### 7. **07_token_tracking.py** - Token & Cost Tracking
- Monitor token usage
- Track API costs
- Usage statistics

#### 8. **08_error_handling.py** - Error Handling
- Handle API errors
- Retry logic
- Graceful fallbacks

### Integration Examples

#### 9. **09_provider_fallback.py** - Multi-Provider Fallback
- Try primary provider
- Fall back to secondary
- Reliability patterns

#### 10. **10_openclaw_integration.py** - Openclaw Integration
- Integrate with Openclaw
- Specialized workflows

#### 11. **11_langchain_integration.py** - LangChain Integration
- Use with LangChain
- Chain operations
- Complex workflows

#### 12. **12_vision_models.py** - Vision/Multimodal
- Image analysis
- Multi-modal inputs
- Vision capabilities

#### 13. **13_function_calling.py** - Function Calling
- Tool use / function calling
- Structured outputs
- Complex reasoning

## Client Classes

### ClaudeClient
```python
from socratic_nexus.clients import ClaudeClient

client = ClaudeClient(
    api_key="sk-ant-...",
    model="claude-3-5-sonnet-20241022"  # or claude-3-5-haiku-20241022
)

response = client.generate_response("Your prompt here")
```

**Models:**
- `claude-3-5-sonnet-20241022` (fastest, most capable)
- `claude-3-5-haiku-20241022` (smaller, faster)
- `claude-3-opus-20250219` (most powerful)

**Methods:**
- `generate_response()` - General text generation
- `generate_code()` - Code generation
- `extract_insights()` - Extract key insights
- `generate_documentation()` - Generate docs
- `generate_business_plan()` - Business planning
- `generate_curriculum()` - Learning design
- And 3 more specialized methods

### OpenAIClient
```python
from socratic_nexus.clients import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",
    model="gpt-4-turbo"
)

response = client.generate_response("Your prompt here")
```

**Models:**
- `gpt-4-turbo` (latest, most capable)
- `gpt-4`
- `gpt-3.5-turbo`

**Methods:**
- `generate_response()` - Text generation
- `generate_code()` - Code generation
- `extract_insights()` - Extract insights

### GoogleClient
```python
from socratic_nexus.clients import GoogleClient

client = GoogleClient(
    api_key="AIza-...",
    model="gemini-pro"
)

response = client.generate_response("Your prompt here")
```

**Models:**
- `gemini-pro` - Standard model
- `gemini-pro-vision` - With vision capability

### OllamaClient
```python
from socratic_nexus.clients import OllamaClient

client = OllamaClient(
    model="llama2",
    base_url="http://localhost:11434"
)

response = client.generate_response("Your prompt here")
```

**Models:**
- `llama2` - Meta's Llama
- `mistral` - Mistral AI
- `neural-chat` - Intel's Neural Chat
- And many more...

## Async/Await Usage

All clients support async methods:

```python
import asyncio
from socratic_nexus.clients import ClaudeClient

async def main():
    client = ClaudeClient(api_key="sk-ant-...")

    # Single async call
    response = await client.generate_response_async("prompt")
    print(response)

    # Parallel calls
    responses = await asyncio.gather(
        client.generate_response_async("prompt 1"),
        client.generate_response_async("prompt 2"),
        client.generate_response_async("prompt 3"),
    )
    for response in responses:
        print(response)

asyncio.run(main())
```

## Configuration

### Environment Variables
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza-..."
```

### In Code
```python
from socratic_nexus.clients import ClaudeClient

client = ClaudeClient(
    api_key="sk-ant-...",  # Or from env: os.getenv("ANTHROPIC_API_KEY")
    model="claude-3-5-haiku-20241022",
    timeout=30,
    max_tokens=1024,
    temperature=0.7
)
```

## Best Practices

1. **Use environment variables** for API keys
2. **Implement error handling** for production code
3. **Use async** for concurrent requests
4. **Cache responses** to save on API costs
5. **Set appropriate timeouts**
6. **Handle rate limits** gracefully

## Troubleshooting

### Missing API Key
```
Error: API key is required
Solution: Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY
```

### Ollama Connection Error
```
Error: Could not connect to http://localhost:11434
Solution: Make sure Ollama is running: ollama serve
```

### Import Error
```
Error: No module named 'socratic_nexus'
Solution: Install it: pip install socratic-nexus
```

## More Examples

For more advanced patterns, see:
- `08_error_handling.py` - Comprehensive error handling
- `09_provider_fallback.py` - Multi-provider patterns
- `11_langchain_integration.py` - LangChain integration
- `12_vision_models.py` - Image analysis
- `13_function_calling.py` - Tool use

## Contributing

Found a bug or want to add an example?
- [GitHub Issues](https://github.com/Nireus79/Socratic-nexus/issues)
- [GitHub Discussions](https://github.com/Nireus79/Socratic-nexus/discussions)

## License

MIT License - See LICENSE file
