# Support & Help

## Getting Help

### 📚 Documentation
- **[README](README.md)** - Overview and quick start
- **[Quick Start Guide](docs/quickstart.md)** - 5-minute setup
- **[Providers Guide](docs/providers.md)** - Provider-specific setup
- **[API Reference](docs/api-reference.md)** - Complete API documentation
- **[Examples](examples/)** - 9 working examples

### 💬 Community

**GitHub Discussions** (Recommended)
- Ask questions: [Discussions](https://github.com/Nireus79/socrates-nexus/discussions)
- Share ideas
- Get help from community
- Browse answered questions

**GitHub Issues**
- Report bugs: [Issues](https://github.com/Nireus79/socrates-nexus/issues)
- Feature requests
- Use issue templates for faster response

### 🐛 Reporting Issues

When reporting an issue, include:
1. **Python version**: `python --version`
2. **Package version**: `pip show socrates-nexus`
3. **OS**: Windows/macOS/Linux
4. **Error message**: Full traceback
5. **Minimal reproduction**: Code that reproduces the issue
6. **Expected behavior**: What should happen
7. **Actual behavior**: What actually happens

**Example issue:**
```
## Description
Chat requests are failing with timeout errors

## Python/Environment
- Python 3.11.0
- socrates-nexus==0.1.0
- macOS 13.5

## Code
```python
from socrates_nexus import LLMClient
client = LLMClient(provider="anthropic", model="claude-opus", api_key="...")
response = client.chat("Hello")  # Times out after 30s
```

## Error: TimeoutError: Request timed out after 30 seconds
```

## Common Issues

### API Key Issues

**"Invalid API key"**
- Verify key is correct: `echo $ANTHROPIC_API_KEY`
- Check key has proper permissions
- Ensure no trailing spaces: `api_key="key ".strip()`
- Try a new key from provider console

**"API key not found"**
```python
# ❌ Wrong
client = LLMClient(provider="anthropic", model="claude-opus")

# ✅ Right
import os
client = LLMClient(
    provider="anthropic",
    model="claude-opus",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
```

### Connection Issues

**"Connection refused"**
- Check internet connection
- Verify API endpoint is accessible
- Check firewall/proxy settings
- Try a different network

**"Timeout error"**
```python
# Increase timeout
from socrates_nexus import LLMClient, LLMConfig

config = LLMConfig(
    provider="anthropic",
    model="claude-opus",
    api_key="...",
    timeout=120,  # 2 minutes
)
client = LLMClient(config=config)
```

### Provider-Specific Issues

**Anthropic/Claude**
- [Anthropic Documentation](https://docs.anthropic.com)
- [Anthropic Support](https://support.anthropic.com)
- Rate limit: 1000 requests/hour (free tier)

**OpenAI/GPT**
- [OpenAI Documentation](https://platform.openai.com/docs)
- [OpenAI Support](https://help.openai.com)
- Rate limit: Varies by plan

**Google/Gemini**
- [Google AI Documentation](https://ai.google.dev)
- [Google AI Support](https://support.google.com/ai)
- Rate limit: Free tier available

**Ollama (Local)**
- Verify Ollama is running: `curl http://localhost:11434`
- Check model exists: `ollama list`
- Pull model: `ollama pull llama2`

### Performance Issues

**Slow responses**
- Check API provider status page
- Try a different model
- Verify network speed
- Increase timeout

**High costs**
- Use cheaper models (Claude Haiku instead of Opus)
- Cache responses: `cache_responses=True`
- Use shorter prompts
- Monitor usage: `client.get_usage_stats()`

### Integration Issues

**"Module not found: socrates_nexus"**
```bash
# Install the package
pip install socrates-nexus

# Or install in development mode
pip install -e .
```

**"ImportError: cannot import X"**
```python
# Check what's available
from socrates_nexus import LLMClient, AsyncLLMClient  # ✅

# Not recommended
from socrates_nexus.providers import anthropic  # ❌ Use LLMClient instead
```

## Frequently Asked Questions

### Q: Which model should I use?
**A:** Start with Claude Haiku (fastest/cheapest). Upgrade to Sonnet or GPT-4 if you need better quality.

### Q: How do I switch providers?
**A:** Change the `provider` parameter - same code works for all:
```python
client = LLMClient(provider="openai", model="gpt-4", api_key="...")
```

### Q: Does it work with local models?
**A:** Yes! Use Ollama:
```python
client = LLMClient(provider="ollama", model="llama2")
```

### Q: How do I reduce costs?
**A:**
1. Use cheaper models (Haiku vs Opus)
2. Cache responses
3. Use shorter prompts
4. Use local models (Ollama)

### Q: Can I use it with async?
**A:** Yes:
```python
from socrates_nexus import AsyncLLMClient
import asyncio

async def main():
    client = AsyncLLMClient(provider="anthropic", ...)
    response = await client.chat("Hello")

asyncio.run(main())
```

### Q: How do I stream responses?
**A:** Use the `stream()` method:
```python
client.stream("Write a poem", on_chunk=print)
```

### Q: Is response caching enabled by default?
**A:** Yes, with 5-minute TTL. Disable with:
```python
config = LLMConfig(..., cache_responses=False)
```

### Q: What Python versions are supported?
**A:** Python 3.8+ (tested on 3.8, 3.9, 3.10, 3.11, 3.12)

## Getting More Help

### Professional Support
For enterprise support:
- Contact: [Nireus79@github.com](mailto:Nireus79@github.com)
- Organization support available

### Real-time Chat
- Discord: (Coming soon)
- Slack: (Coming soon)

## Give Feedback

We'd love to hear from you!
- ⭐ Star on GitHub if you find it useful
- 💬 Share feedback in Discussions
- 🐛 Report issues
- 📝 Contribute docs/examples
- 📋 Suggest features

## Useful Resources

### Official Documentation
- [Anthropic Claude](https://docs.anthropic.com)
- [OpenAI API](https://platform.openai.com/docs)
- [Google Generative AI](https://ai.google.dev)
- [Ollama](https://ollama.ai)

### Related Projects
- [LangChain](https://langchain.com) - AI framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Another LLM client
- [Vercel AI SDK](https://github.com/vercel/ai) - Streaming LLM

### Learning Resources
- [AI Fundamentals](https://github.com/Nireus79/Socrates) - Parent project
- [LLM Basics](https://platform.openai.com/docs/guides/gpt)
- [Prompt Engineering](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)

---

**Last Updated:** March 9, 2026
**Response Time:** 24-48 hours for discussions/issues
**Maintained By:** Nireus79 Team
