"""
Example 1: Using Socrates Nexus with Claude (Anthropic).

This example shows how to use the Claude API with Socrates Nexus.
"""

import os
from socrates_nexus import LLMClient

# Initialize client with Claude
client = LLMClient(
    provider="anthropic",
    model="claude-opus",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Chat example
print("Coming soon: Chat example with Claude")
print("This shows automatic retries, token tracking, and streaming support")
