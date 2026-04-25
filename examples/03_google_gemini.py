"""
Example 3: Google Gemini Usage with Socratic Nexus

Demonstrates how to use Socratic Nexus with Google's Gemini models.
"""

import os
from socratic_nexus.clients import GoogleClient

# Initialize Google Gemini client
client = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY", "AIza-..."),
    model="gemini-pro"  # or gemini-pro-vision
)

print("=" * 60)
print("GOOGLE GEMINI - BASIC USAGE")
print("=" * 60)

# Simple response generation
prompt = "List 3 benefits of cloud computing in bullet points"
response = client.generate_response(prompt)

print(f"\nPrompt: {prompt}")
print(f"Model: {client.model}")
print(f"\nResponse:\n{response}")
print("\n" + "=" * 60)

# Generate code
if False:
    code = client.generate_code(
        "Create a REST API endpoint in Python"
    )
    print(f"\nGenerated Code:\n{code}")

# Extract insights
if False:
    insights = client.extract_insights(
        "What are the advantages of microservices architecture?"
    )
    print(f"\nExtracted Insights:\n{insights}")
