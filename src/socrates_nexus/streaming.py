"""Streaming helpers for Socrates Nexus."""

from typing import Callable, Optional


class StreamBuffer:
    """Buffer for streaming responses."""

    def __init__(self, on_chunk: Optional[Callable[[str], None]] = None):
        """
        Initialize stream buffer.

        Args:
            on_chunk: Optional callback for each chunk
        """
        self.on_chunk = on_chunk
        self.buffer = ""
        self.complete = False

    def add_chunk(self, chunk: str) -> None:
        """Add a chunk to the buffer."""
        self.buffer += chunk
        if self.on_chunk:
            self.on_chunk(chunk)

    def get_complete(self) -> str:
        """Get the complete buffered content."""
        return self.buffer

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = ""
