"""Tests for exception classes."""

import pytest
from socratic_nexus.exceptions import APIError


def test_api_error_creation():
    """Test APIError creation."""
    error = APIError("Test error message")
    assert str(error) == "Test error message"


def test_api_error_with_error_type():
    """Test APIError with error_type."""
    error = APIError("Test error", error_type="RATE_LIMIT")
    assert str(error) == "Test error"
    # Check if error_type is stored (if the implementation stores it)
    # This depends on how APIError is implemented


def test_api_error_inheritance():
    """Test that APIError is an Exception."""
    error = APIError("Test")
    assert isinstance(error, Exception)


def test_api_error_can_be_raised():
    """Test that APIError can be raised and caught."""
    with pytest.raises(APIError):
        raise APIError("Test error")


def test_api_error_message_preserved():
    """Test that error message is preserved."""
    message = "Detailed error message with context"
    with pytest.raises(APIError) as exc_info:
        raise APIError(message)
    assert message in str(exc_info.value)
