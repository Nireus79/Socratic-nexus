"""Exceptions for Socratic Nexus LLM client library."""

from __future__ import annotations


class APIError(Exception):
    """Base exception for API-related errors."""

    def __init__(self, message: str, error_type: str = "unknown", **kwargs):
        """Initialize APIError."""
        self.message = message
        self.error_type = error_type
        self.details = kwargs
        super().__init__(message)


__all__ = ["APIError"]
