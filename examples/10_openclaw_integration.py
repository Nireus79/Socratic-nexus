"""
Example 10: Integration Patterns

Demonstrates how to integrate Socratic Nexus clients into larger applications.
Shows patterns for:
- Custom wrappers
- Agent integration
- Event handling
"""

import os
from socratic_nexus.clients import ClaudeClient

print("=" * 60)
print("INTEGRATION PATTERN - CUSTOM WRAPPER")
print("=" * 60)


class MyLLMService:
    """Custom wrapper around Socratic Nexus clients."""

    def __init__(self, provider_name: str = "claude"):
        """Initialize with a specific provider."""
        self.provider_name = provider_name

        if provider_name == "claude":
            self.client = ClaudeClient(
                api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-...")
            )
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    def ask(self, question: str) -> str:
        """Simple interface to ask questions."""
        return self.client.generate_response(question)

    def code(self, spec: str) -> str:
        """Generate code from specification."""
        return self.client.generate_code(spec)

    def insights(self, text: str) -> str:
        """Extract insights from text."""
        return self.client.extract_insights(text)


# Usage
print("\nCreating custom LLM service...")
service = MyLLMService()

print("\n1. Asking a question:")
answer = service.ask("What is machine learning?")
print(f"   {answer[:80]}...")

print("\n2. Generating code:")
code = service.code("Function to calculate fibonacci")
print(f"   {code[:80]}...")

print("\n3. Extracting insights:")
insights = service.insights("The future of AI is uncertain but promising")
print(f"   {insights[:80]}...")

print("\n" + "=" * 60)
print("INTEGRATION PATTERN COMPLETE")
print("=" * 60)
