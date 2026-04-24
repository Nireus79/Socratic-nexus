"""Event types for Socratic Nexus LLM client library."""

from __future__ import annotations

from enum import Enum


class EventType(Enum):
    """Event types emitted by the library."""

    TOKEN_USAGE = "token_usage"
    LOG_ERROR = "log_error"
    LOG_WARNING = "log_warning"
    LOG_INFO = "log_info"
    REQUEST_STARTED = "request_started"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"


__all__ = ["EventType"]
