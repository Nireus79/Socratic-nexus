"""
Example 1: Basic Claude Usage with Socratic Nexus

Demonstrates the simplest way to use Socratic Nexus with Anthropic's Claude.
"""

import os
from socratic_nexus.clients import ClaudeClient

# Initialize Claude client
client = ClaudeClient(
    api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."),
    model="claude-3-5-sonnet-20241022"  # or claude-3-5-haiku-20241022
)

# Generate a simple response
prompt = "What is machine learning? Explain in 2 sentences."
response = client.generate_response(prompt)

print("=" * 60)
print("CLAUDE - BASIC USAGE")
print("=" * 60)
print(f"\nPrompt: {prompt}")
print(f"\nResponse:\n{response}")
print("\n" + "=" * 60)

# For orchestrator integration
if False:  # Set to True if using with AgentOrchestrator
    from socratic_nexus.models import ProjectContext

    # Use with project context
    project = ProjectContext(
        name="My Project",
        description="Learning about ML"
    )
    insights = client.extract_insights(prompt, project=project)
    print(f"\nExtracted Insights:\n{insights}")
