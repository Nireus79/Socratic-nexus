"""
Example 7: Token Tracking with Socratic Nexus

Demonstrates basic token usage tracking across requests.
Note: Full token tracking with cost calculation requires orchestrator integration.
"""

import os
from socratic_nexus.clients import ClaudeClient

print("=" * 60)
print("TOKEN TRACKING - RESPONSE ANALYSIS")
print("=" * 60)

# Create client
client = ClaudeClient(
    api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."),
    model="claude-3-5-haiku-20241022"
)

# Send multiple requests and track usage
queries = [
    "What is machine learning?",
    "Explain neural networks",
    "What is deep learning?",
]

print("\nSending 3 requests...\n")

total_tokens = 0
total_cost = 0.0
request_count = 0

for i, query in enumerate(queries, 1):
    response = client.generate_response(query)
    # Note: Basic clients don't return token info; it's available via orchestrator
    print(f"Request {i}: '{query[:30]}...'")
    print(f"Response length: {len(response)} characters\n")
    request_count += 1

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total requests sent: {request_count}")
print("\nNote: Detailed token tracking is available when using:")
print("  - AgentOrchestrator with clients")
print("  - Custom event listeners")
print("  - Integration with monitoring systems")
