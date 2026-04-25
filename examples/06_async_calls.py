"""
Example 6: Async/Await and Concurrent Requests

Demonstrates asynchronous usage and parallel requests with Socratic Nexus.
"""

import asyncio
import os
from socratic_nexus.clients import ClaudeClient, OpenAIClient

print("=" * 60)
print("ASYNC - EXAMPLE 1: Single Async Request")
print("=" * 60)


async def single_async_request():
    """Simple async request to Claude."""
    client = ClaudeClient(
        api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."),
        model="claude-3-5-haiku-20241022"
    )

    response = await client.generate_response_async("What is the speed of light?")
    print(f"Response: {response}")


asyncio.run(single_async_request())

# Example 2: Parallel requests to same provider
print("\n" + "=" * 60)
print("ASYNC - EXAMPLE 2: Parallel Requests (Same Provider)")
print("=" * 60)


async def parallel_same_provider():
    """Send multiple concurrent requests to the same provider."""
    client = ClaudeClient(
        api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."),
        model="claude-3-5-haiku-20241022"
    )

    # Launch all requests concurrently
    responses = await asyncio.gather(
        client.generate_response_async("What is Python?"),
        client.generate_response_async("What is JavaScript?"),
        client.generate_response_async("What is Rust?"),
    )

    for i, response in enumerate(responses, 1):
        print(f"\nResponse {i}:\n{response[:100]}...")


asyncio.run(parallel_same_provider())

# Example 3: Parallel requests to different providers
print("\n" + "=" * 60)
print("ASYNC - EXAMPLE 3: Parallel Requests (Different Providers)")
print("=" * 60)


async def parallel_different_providers():
    """Send concurrent requests to different providers."""
    claude = ClaudeClient(
        api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."),
        model="claude-3-5-haiku-20241022"
    )
    openai = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY", "sk-..."),
        model="gpt-3.5-turbo"
    )

    prompt = "Explain machine learning in one sentence"

    # Get responses from both providers concurrently
    claude_response, openai_response = await asyncio.gather(
        claude.generate_response_async(prompt),
        openai.generate_response_async(prompt),
    )

    print(f"\nClaude:\n{claude_response[:100]}...")
    print(f"\nOpenAI:\n{openai_response[:100]}...")


asyncio.run(parallel_different_providers())
