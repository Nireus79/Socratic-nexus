"""Tests for event types and event handling."""

from socratic_nexus.events import EventType


def test_event_type_enum():
    """Test that EventType is properly defined."""
    assert hasattr(EventType, "TOKEN_USAGE")
    assert hasattr(EventType, "LOG_ERROR")
    assert hasattr(EventType, "LOG_WARNING")
    assert hasattr(EventType, "LOG_INFO")
    assert hasattr(EventType, "REQUEST_STARTED")
    assert hasattr(EventType, "REQUEST_COMPLETED")
    assert hasattr(EventType, "REQUEST_FAILED")


def test_event_type_values():
    """Test that EventType enum values exist."""
    # Test that we can access enum values
    token_usage = EventType.TOKEN_USAGE
    assert token_usage is not None

    log_error = EventType.LOG_ERROR
    assert log_error is not None

    request_completed = EventType.REQUEST_COMPLETED
    assert request_completed is not None


def test_event_type_comparison():
    """Test EventType enum comparison."""
    assert EventType.TOKEN_USAGE == EventType.TOKEN_USAGE
    assert EventType.TOKEN_USAGE != EventType.LOG_ERROR


def test_event_type_string_representation():
    """Test EventType string representation."""
    event_str = str(EventType.TOKEN_USAGE)
    assert "TOKEN_USAGE" in event_str
