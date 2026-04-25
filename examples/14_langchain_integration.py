"""
Example 14: LangChain Integration

Demonstrates how to use Socratic Nexus clients with LangChain chains,
agents, and other LangChain components.

Installation:
    pip install socratic-nexus langchain
"""

import os
from socratic_nexus.clients import ClaudeClient
from socratic_nexus.integrations.langchain import SocratesNexusLLM

print("=" * 60)
print("LANGCHAIN INTEGRATION - BASIC EXAMPLE")
print("=" * 60)

# Create a Socratic Nexus client
client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."))

# Wrap it with the LangChain integration
llm = SocratesNexusLLM(client=client, temperature=0.7)

print("\n1. Direct LLM call:")
print("-" * 40)
response = llm("What is machine learning?")
print(f"Response: {response[:150]}...\n")

# Example with LangChain chains (requires langchain)
print("2. Using with LangChain chains:")
print("-" * 40)

try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in simple terms for beginners.",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(topic="Neural Networks")
    print(f"Chain result: {result[:150]}...\n")

except ImportError:
    print("LangChain not installed. Install with: pip install langchain\n")

# Example with multiple providers
print("3. Using different clients:")
print("-" * 40)

try:
    from socratic_nexus.clients import OpenAIClient, GoogleClient

    openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY", "sk-..."))
    openai_llm = SocratesNexusLLM(client=openai_client)
    print(f"Created OpenAI LLM wrapper: {openai_llm._llm_type}")

    google_client = GoogleClient(api_key=os.getenv("GOOGLE_API_KEY", "AIza-..."))
    google_llm = SocratesNexusLLM(client=google_client)
    print(f"Created Google LLM wrapper: {google_llm._llm_type}\n")

except ImportError as e:
    print(f"Provider not available: {e}\n")

print("=" * 60)
print("LANGCHAIN INTEGRATION COMPLETE")
print("=" * 60)

print("\nKey features:")
print("- Use Socratic Nexus clients in LangChain chains")
print("- Support for all client types (Claude, OpenAI, Google, Ollama)")
print("- Asynchronous support for async LangChain operations")
print("- Token tracking and cost calculation")

print("\nFor more information:")
print("- LangChain docs: https://python.langchain.com")
print("- Socratic Nexus: https://github.com/Nireus79/Socratic-nexus")
