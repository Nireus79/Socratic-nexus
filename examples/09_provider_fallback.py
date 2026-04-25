"""
Example 9: Provider Fallback and Multi-Model Resilience

Demonstrates fallback strategies when a provider fails:
- Sequential fallback: try provider 1, then provider 2
- Parallel fallback: try multiple providers concurrently
"""

import asyncio
import os
from socratic_nexus.clients import ClaudeClient, OpenAIClient, GoogleClient

print("=" * 60)
print("PROVIDER FALLBACK - SEQUENTIAL APPROACH")
print("=" * 60)


def safe_chat_sequential(message: str) -> dict:
    """Try each provider in sequence until one succeeds."""

    providers = [
        ("Claude", ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."))),
        ("OpenAI", OpenAIClient(api_key=os.getenv("OPENAI_API_KEY", "sk-..."))),
        ("Google", GoogleClient(api_key=os.getenv("GOOGLE_API_KEY", "AIza-..."))),
    ]

    for name, client in providers:
        try:
            print(f"\nAttempting with {name}...")
            response = client.generate_response(message)
            print(f"SUCCESS with {name}!")
            return {
                "message": message,
                "response": response,
                "provider": name,
                "success": True,
            }
        except Exception as e:
            print(f"FALLBACK: {name} failed ({type(e).__name__})")
            continue

    return {
        "message": message,
        "response": None,
        "provider": None,
        "success": False,
        "error": "All providers failed",
    }


# Test sequential fallback
result = safe_chat_sequential("What is quantum computing?")

if result["success"]:
    print(f"\n✓ Success using {result['provider']}")
    print(f"Response: {result['response'][:100]}...")
else:
    print(f"\n✗ All providers failed")


# Example 2: Parallel fallback
print("\n" + "=" * 60)
print("PROVIDER FALLBACK - PARALLEL APPROACH")
print("=" * 60)


async def safe_chat_parallel(message: str) -> dict:
    """Try multiple providers concurrently, use first successful."""

    async def try_provider(name, client):
        try:
            response = await client.generate_response_async(message)
            return {"name": name, "success": True, "response": response}
        except Exception as e:
            return {"name": name, "success": False, "error": str(e)}

    providers = [
        ("Claude", ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."))),
        ("OpenAI", OpenAIClient(api_key=os.getenv("OPENAI_API_KEY", "sk-..."))),
        ("Google", GoogleClient(api_key=os.getenv("GOOGLE_API_KEY", "AIza-..."))),
    ]

    # Race all providers
    tasks = [try_provider(name, client) for name, client in providers]
    results = await asyncio.gather(*tasks)

    # Use first successful
    for result in results:
        if result["success"]:
            print(f"\n✓ First successful: {result['name']}")
            return {
                "message": message,
                "response": result['response'],
                "provider": result['name'],
                "success": True,
            }

    print(f"\n✗ All providers failed")
    return {
        "message": message,
        "response": None,
        "provider": None,
        "success": False,
    }


# Test parallel fallback (uncomment to run)
if False:
    result = asyncio.run(safe_chat_parallel("What is AI?"))
