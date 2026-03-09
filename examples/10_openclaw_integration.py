"""
Example: Using Socrates Nexus with Openclaw

Demonstrates how to use Socrates Nexus as an Openclaw Skill for multi-provider LLM support.

Installation:
    pip install socrates-nexus[openclaw]
"""

from socrates_nexus.integrations.openclaw import NexusLLMSkill


def example_basic_usage():
    """Basic Openclaw skill usage."""
    print("=== Basic Openclaw Skill Usage ===\n")

    skill = NexusLLMSkill(provider="anthropic", model="claude-opus")

    response = skill.query("What is machine learning in one sentence?")
    print(f"Query: 'What is machine learning in one sentence?'")
    print(f"Response: {response.content}")
    print(f"Cost: ${response.usage.cost_usd:.6f}")
    print(f"Tokens: {response.usage.total_tokens}\n")


def example_provider_switching():
    """Switch between providers."""
    print("=== Provider Switching ===\n")

    skill = NexusLLMSkill(provider="anthropic", model="claude-opus")

    # Query with Claude
    print("1. Query with Claude Opus:")
    response1 = skill.query("What is AI?")
    print(f"   {response1.content[:100]}...\n")

    # Switch to OpenAI
    print("2. Switch to GPT-4 and query:")
    skill.switch_provider("openai", "gpt-4")
    response2 = skill.query("What is AI?")
    print(f"   {response2.content[:100]}...\n")

    # Switch to Google
    print("3. Switch to Gemini and query:")
    skill.switch_provider("google", "gemini-pro")
    response3 = skill.query("What is AI?")
    print(f"   {response3.content[:100]}...\n")


def example_usage_tracking():
    """Track token usage across multiple queries."""
    print("=== Usage Tracking ===\n")

    skill = NexusLLMSkill(provider="anthropic", model="claude-opus")

    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    for query in queries:
        response = skill.query(query)
        print(f"Query: {query}")
        print(f"  Tokens: {response.usage.total_tokens}, Cost: ${response.usage.cost_usd:.6f}")

    stats = skill.get_usage_stats()
    print(f"\nTotal Stats:")
    print(f"  Requests: {stats['total_requests']}")
    print(f"  Total Cost: ${stats['total_cost_usd']:.6f}")
    print(f"  Total Tokens: {stats['total_input_tokens'] + stats['total_output_tokens']}\n")


def example_callback_tracking():
    """Use callbacks for real-time tracking."""
    print("=== Callback Tracking ===\n")

    def track_usage(usage):
        print(f"  [Usage] Tokens: {usage.total_tokens}, Cost: ${usage.cost_usd:.6f}")

    skill = NexusLLMSkill(provider="anthropic", model="claude-opus")
    skill.add_usage_callback(track_usage)

    print("Query 1:")
    skill.query("What is machine learning?")

    print("\nQuery 2:")
    skill.query("What is deep learning?")

    print("\nQuery 3:")
    skill.query("What is reinforcement learning?\n")


def example_streaming():
    """Stream responses for real-time output."""
    print("=== Streaming Responses ===\n")

    skill = NexusLLMSkill(provider="anthropic", model="claude-opus")

    def on_chunk(chunk):
        print(chunk, end="", flush=True)

    print("Query: 'Write a haiku about programming'")
    print("\nResponse:\n")
    response = skill.stream("Write a haiku about programming", on_chunk=on_chunk)
    print(f"\n\nCost: ${response.usage.cost_usd:.6f}\n")


if __name__ == "__main__":
    print("Socrates Nexus - Openclaw Integration Examples\n")
    print("=" * 50 + "\n")

    example_basic_usage()
    example_provider_switching()
    example_usage_tracking()
    example_callback_tracking()
    example_streaming()

    print("=" * 50)
    print("Examples complete!")
