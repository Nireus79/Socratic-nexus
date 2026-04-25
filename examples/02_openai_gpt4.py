"""
Example 2: OpenAI GPT-4 Usage with Socratic Nexus

Demonstrates how to use Socratic Nexus with OpenAI's GPT-4 models.
"""

import os
from socratic_nexus.clients import OpenAIClient

# Initialize OpenAI client with GPT-4
client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY", "sk-..."),
    model="gpt-4-turbo"  # or gpt-4, gpt-3.5-turbo
)

# Generate a response
prompt = "Explain quantum computing in simple terms."
response = client.generate_response(prompt)

print("=" * 60)
print("OPENAI GPT-4 - BASIC USAGE")
print("=" * 60)
print(f"\nPrompt: {prompt}")
print(f"Model: {client.model}")
print(f"\nResponse:\n{response}")
print("\n" + "=" * 60)

# Generate code
if False:
    code_spec = "Create a function to calculate the Fibonacci sequence"
    code = client.generate_code(code_spec)
    print(f"\nGenerated Code:\n{code}")

# Extract insights
if False:
    insights = client.extract_insights(
        "What are the key principles of object-oriented programming?"
    )
    print(f"\nExtracted Insights:\n{insights}")
