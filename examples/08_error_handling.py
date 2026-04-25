"""
Example 8: Error Handling and Resilience

Demonstrates how to handle errors gracefully with Socratic Nexus.
"""

import os
from socratic_nexus.clients import ClaudeClient, OpenAIClient

print("=" * 60)
print("ERROR HANDLING - GRACEFUL DEGRADATION")
print("=" * 60)


def test_with_invalid_key():
    """Test with invalid API key - handle gracefully."""
    print("\n1. Testing with invalid API key...")
    try:
        client = ClaudeClient(api_key="invalid-key-123")
        response = client.generate_response("Hello")
    except Exception as e:
        print(f"   [HANDLED] Error: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")


def test_with_timeout():
    """Test timeout handling."""
    print("\n2. Testing with custom timeout...")
    try:
        client = ClaudeClient(
            api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."),
            timeout=1  # Very short timeout for demo
        )
        # This would timeout
        response = client.generate_response("Write a very long essay...")
    except Exception as e:
        print(f"   [HANDLED] Timeout: {type(e).__name__}")


def test_provider_fallback():
    """Fallback to another provider if first fails."""
    print("\n3. Testing provider fallback...")

    providers = [
        ("Claude", lambda: ClaudeClient(
            api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-...")
        )),
        ("OpenAI", lambda: OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY", "sk-...")
        )),
    ]

    prompt = "What is AI?"

    for name, create_client in providers:
        try:
            client = create_client()
            response = client.generate_response(prompt)
            print(f"   [SUCCESS] Used {name} provider")
            print(f"   Response: {response[:60]}...")
            break
        except Exception as e:
            print(f"   [FALLBACK] {name} failed: {type(e).__name__}")


def test_validation():
    """Test input validation."""
    print("\n4. Testing input validation...")

    client = ClaudeClient(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-..."))

    try:
        # Empty prompt
        response = client.generate_response("")
        print("   [HANDLED] Empty prompt accepted")
    except Exception as e:
        print(f"   [VALIDATION] Empty prompt rejected: {e}")

    try:
        # None prompt
        response = client.generate_response(None)
    except Exception as e:
        print(f"   [VALIDATION] None prompt rejected: {type(e).__name__}")


print("\nRunning error handling tests...\n")

test_with_invalid_key()
test_with_timeout()
test_provider_fallback()
test_validation()

print("\n" + "=" * 60)
print("ERROR HANDLING COMPLETE")
print("=" * 60)
