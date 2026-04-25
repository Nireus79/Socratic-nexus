"""
Example 4: Ollama Local LLM with Socratic Nexus

Demonstrates running models locally with Ollama (no API key needed).

Prerequisites:
  1. Install Ollama from https://ollama.ai
  2. Run: ollama pull llama2
  3. Run: ollama serve (starts on http://localhost:11434)
"""

from socratic_nexus.clients import OllamaClient

print("=" * 60)
print("OLLAMA LOCAL MODEL - NO API KEY REQUIRED")
print("=" * 60)

# Initialize Ollama client (no API key needed)
# Make sure Ollama is running: ollama serve
client = OllamaClient(
    model="llama2",  # or mistral, neural-chat, orca-mini
    base_url="http://localhost:11434",  # Default Ollama URL
)

print("\nSending request to local Ollama model...")

try:
    prompt = "What is the capital of France? Answer in one sentence."
    response = client.generate_response(prompt)

    print(f"\nPrompt: {prompt}")
    print(f"Model: {client.model}")
    print(f"Base URL: {client.base_url}")
    print(f"\nResponse:\n{response}")
    print(f"\nCost: FREE (local models don't incur API costs!)")

except Exception as e:
    print(f"\nError: {e}")
    print("\nMake sure Ollama is running:")
    print("  1. ollama pull llama2")
    print("  2. ollama serve")
    print("\nThen try again!")
