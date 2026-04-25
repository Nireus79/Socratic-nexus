"""
Example 11: LangChain Integration

Demonstrates how to use Socratic Nexus clients with LangChain.

Installation:
    pip install socrates-nexus langchain
"""

import os
from socratic_nexus.clients import ClaudeClient

print("=" * 60)
print("LANGCHAIN INTEGRATION CONCEPT")
print("=" * 60)

# Note: Full LangChain integration would require implementing
# LangChain's LLM interface. This example shows the conceptual approach.


class SimpleChain:
    """Simple chain pattern with Socratic Nexus client."""

    def __init__(self):
        self.client = ClaudeClient(
            api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-...")
        )

    def run(self, prompt_template: str, variables: dict) -> str:
        """Run a simple chain with variable substitution."""
        # Format the template with variables
        formatted_prompt = prompt_template.format(**variables)

        # Get response
        response = self.client.generate_response(formatted_prompt)

        return response


# Example usage
print("\nExample 1: Simple Template Chain")
print("-" * 40)

chain = SimpleChain()

template = "Write a short poem about {topic} in the style of {style}."
variables = {
    "topic": "technology",
    "style": "Shakespeare"
}

result = chain.run(template, variables)
print(f"Prompt: {template.format(**variables)}")
print(f"Result: {result[:100]}...\n")

# Example 2: Sequential operations
print("\nExample 2: Sequential Operations")
print("-" * 40)

client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."))

# Step 1: Generate initial content
content = client.generate_response("Write a summary of quantum computing")
print(f"Step 1 (Generate): {content[:80]}...")

# Step 2: Extract insights
insights = client.extract_insights(content)
print(f"Step 2 (Extract): {insights[:80]}...")

print("\n" + "=" * 60)
print("INTEGRATION DEMONSTRATION COMPLETE")
print("=" * 60)

print("\nNote: For full LangChain integration, you would:")
print("1. Inherit from langchain.llms.base.LLM")
print("2. Implement required methods (_generate, _llm_type)")
print("3. Use with LangChain chains and agents")
print("\nThe Socratic Nexus clients work great with LangChain's")
print("flexible architecture!")
