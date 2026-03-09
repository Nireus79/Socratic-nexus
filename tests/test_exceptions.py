"""Tests for exception hierarchy and error handling."""

import pytest
from socrates_nexus.exceptions import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidAPIKeyError,
    ProviderError,
)


def test_llm_error_base():
    """Test LLMError base exception."""
    error = LLMError(
        message="Test error",
        error_code="TEST_ERROR",
        context={"provider": "test"},
    )

    assert error.message == "Test error"
    assert error.error_code == "TEST_ERROR"
    assert error.context == {"provider": "test"}
    assert "Test error" in str(error)


def test_rate_limit_error():
    """Test RateLimitError with retry_after."""
    error = RateLimitError(
        message="Rate limited",
        retry_after=30,
    )

    assert error.message == "Rate limited"
    assert error.error_code == "RATE_LIMIT"
    assert error.retry_after == 30


def test_rate_limit_error_no_retry_after():
    """Test RateLimitError without retry_after."""
    error = RateLimitError(
        message="Rate limited",
    )

    assert error.retry_after is None


def test_authentication_error():
    """Test AuthenticationError."""
    error = AuthenticationError(
        message="Invalid credentials",
    )

    assert error.message == "Invalid credentials"
    assert error.error_code == "AUTHENTICATION_ERROR"


def test_invalid_api_key_error():
    """Test InvalidAPIKeyError."""
    error = InvalidAPIKeyError(
        message="API key is invalid",
    )

    assert error.message == "API key is invalid"


def test_provider_error():
    """Test ProviderError."""
    error = ProviderError(
        message="Provider error",
        error_code="PROVIDER_ERROR",
        context={"provider": "anthropic"},
    )

    assert error.message == "Provider error"
    assert error.error_code == "PROVIDER_ERROR"


def test_error_inheritance():
    """Test that all specific errors inherit from LLMError."""
    errors = [
        RateLimitError(message="test"),
        AuthenticationError(message="test"),
        InvalidAPIKeyError(message="test"),
        ProviderError(message="test", error_code="TEST"),
    ]

    for error in errors:
        assert isinstance(error, LLMError)


def test_error_catching_by_base_class():
    """Test that errors can be caught by base class."""
    error = InvalidAPIKeyError(message="Invalid key")

    try:
        raise error
    except LLMError as e:
        assert e.message == "Invalid key"


def test_error_catching_by_specific_class():
    """Test that errors can be caught by specific class."""
    error = RateLimitError(message="Rate limited")

    try:
        raise error
    except RateLimitError as e:
        assert e.retry_after is None


def test_error_context_dict():
    """Test that error context is a dict."""
    error = LLMError(
        message="Test",
        error_code="TEST",
        context={"key1": "value1", "key2": "value2"},
    )

    assert isinstance(error.context, dict)
    assert error.context["key1"] == "value1"
    assert error.context["key2"] == "value2"


def test_error_string_representation():
    """Test error string representation."""
    error = LLMError(
        message="Something went wrong",
        error_code="ERROR_CODE",
    )

    error_str = str(error)
    assert "Something went wrong" in error_str
