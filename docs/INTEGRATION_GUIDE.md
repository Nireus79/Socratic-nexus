# Socrates Nexus - Integration Guide

## Openclaw Integration

### Basic Setup

Install with Openclaw support:
```bash
pip install socrates-nexus[openclaw]
```

### Creating an Openclaw Skill

```python
from socrates_nexus.integrations.openclaw import NexusLLMSkill

skill = NexusLLMSkill(provider="anthropic", model="claude-opus")
response = skill.query("What is machine learning?")
print(response)
```

### Switching Providers

```python
skill = NexusLLMSkill(provider="anthropic", model="claude-opus")
response = skill.query("Query 1")

# Switch provider
skill.switch_provider("openai", "gpt-4")
response = skill.query("Query 2")
```

### Configuration Options

```python
skill = NexusLLMSkill(
    provider="anthropic",
    model="claude-opus",
    temperature=0.7,
    max_tokens=1024,
    cache_responses=True,
    retry_attempts=3
)
```

---

## LangChain Integration

### Installation

```bash
pip install socrates-nexus[langchain]
```

### Using with LangChain Chains

```python
from socrates_nexus.integrations.langchain import SocratesNexusLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Create Socrates Nexus LLM
llm = SocratesNexusLLM(
    provider="anthropic",
    model="claude-opus"
)

# Use in LangChain chain
template = "You are a helpful assistant. {query}"
prompt = PromptTemplate(template=template, input_variables=["query"])
chain = LLMChain(llm=llm, prompt=prompt)

# Execute
result = chain.run(query="What is machine learning?")
```

### Using with LangChain Agents

```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=[],  # Add your tools
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("What should I do?")
```

### Multi-Provider Chaining

```python
# Primary LLM
primary_llm = SocratesNexusLLM(provider="anthropic", model="claude-opus")

# Fallback LLM
fallback_llm = SocratesNexusLLM(provider="openai", model="gpt-4")

# Use primary in chain, fallback in error handling
```

---

## Socratic Ecosystem Integration

### With Socratic RAG

```python
from socratic_rag import RAGClient
from socrates_nexus import LLMClient

# RAG retrieves documents
rag = RAGClient()
context = rag.retrieve_context("Your question")

# Socrates Nexus generates answer
llm = LLMClient(provider="anthropic", model="claude-opus")
response = llm.chat(f"Using this context: {context}\n\nAnswer: Your question")
```

### With Socratic Analyzer

```python
from socratic_analyzer import AnalyzerClient
from socratic_analyzer.llm import LLMPoweredAnalyzer
from socrates_nexus import LLMClient

analyzer = AnalyzerClient()
llm = LLMClient(provider="anthropic", model="claude-opus")

llm_analyzer = LLMPoweredAnalyzer(analyzer, llm)
result = llm_analyzer.analyze_with_insights("code.py")
```

### With Socratic Agents

```python
from socratic_agents import CodeGenerator
from socrates_nexus import LLMClient

llm = LLMClient(provider="anthropic", model="claude-opus")
generator = CodeGenerator(llm_client=llm)

result = generator.process({
    "prompt": "Create a sorting algorithm",
    "language": "python"
})
```

---

## Custom Integration Patterns

### Fallback Pattern

```python
class ResilientLLM:
    def __init__(self, providers):
        self.providers = providers
    
    def chat(self, message):
        for config in self.providers:
            try:
                client = LLMClient(**config)
                return client.chat(message)
            except Exception:
                continue
        raise Exception("All providers failed")

# Usage
resilient = ResilientLLM([
    {"provider": "anthropic", "model": "claude-opus"},
    {"provider": "openai", "model": "gpt-4"},
])
response = resilient.chat("Your question")
```

### Cost Optimization

```python
class CostOptimizedLLM:
    def __init__(self):
        self.cheap = LLMClient(provider="openai", model="gpt-3.5-turbo")
        self.expensive = LLMClient(provider="anthropic", model="claude-opus")
    
    def chat(self, message):
        # Try cheap first
        response = self.cheap.chat(message)
        
        # If response quality low, use expensive
        if not self.is_good_response(response):
            response = self.expensive.chat(message)
        
        return response
```

### Monitoring & Logging

```python
class MonitoredLLM:
    def __init__(self, llm_config):
        self.client = LLMClient(**llm_config)
        self.stats = {"requests": 0, "total_cost": 0.0}
    
    def chat(self, message):
        response = self.client.chat(message)
        self.stats["requests"] += 1
        self.stats["total_cost"] += response.usage.cost_usd
        return response
    
    def get_stats(self):
        return self.stats
```

---

## Framework Compatibility

### Flask Integration

```python
from flask import Flask, jsonify
from socrates_nexus import LLMClient

app = Flask(__name__)
llm_client = LLMClient(provider="anthropic", model="claude-opus")

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    response = llm_client.chat(message)
    return jsonify({
        "response": response.content,
        "cost": response.usage.cost_usd
    })
```

### FastAPI Integration

```python
from fastapi import FastAPI
from socrates_nexus import AsyncLLMClient

app = FastAPI()
llm_client = AsyncLLMClient(provider="anthropic", model="claude-opus")

@app.post("/chat")
async def chat(message: str):
    response = await llm_client.chat(message)
    return {
        "response": response.content,
        "cost": response.usage.cost_usd
    }
```
