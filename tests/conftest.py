"""Pytest configuration for Socrates Nexus tests."""

import pytest
import os


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment."""
    return os.getenv("ANTHROPIC_API_KEY")


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture
def google_api_key():
    """Get Google API key from environment."""
    return os.getenv("GOOGLE_API_KEY")
