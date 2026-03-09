"""Tests for streaming functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, call

from socrates_nexus.streaming import StreamHandler, StreamBuffer, AsyncStreamHandler


class TestStreamHandler:
    """Test StreamHandler class."""

    def test_initialization_without_callback(self):
        """Test handler initialization without callback."""
        handler = StreamHandler()
        assert handler.on_chunk is None
        assert handler.chunks == []
        assert handler.complete is False

    def test_initialization_with_callback(self):
        """Test handler initialization with callback."""
        callback = Mock()
        handler = StreamHandler(on_chunk=callback)
        assert handler.on_chunk == callback
        assert handler.chunks == []

    def test_handle_single_chunk(self):
        """Test handling a single chunk."""
        handler = StreamHandler()
        handler.handle_chunk("Hello")
        assert handler.chunks == ["Hello"]

    def test_handle_multiple_chunks(self):
        """Test handling multiple chunks."""
        handler = StreamHandler()
        handler.handle_chunk("Hello")
        handler.handle_chunk(" ")
        handler.handle_chunk("World")
        assert handler.chunks == ["Hello", " ", "World"]

    def test_handle_empty_chunk_not_added(self):
        """Test that empty chunks are not added."""
        handler = StreamHandler()
        handler.handle_chunk("")
        handler.handle_chunk("Hello")
        handler.handle_chunk("")
        assert handler.chunks == ["Hello"]

    def test_get_complete_response(self):
        """Test getting complete response."""
        handler = StreamHandler()
        handler.handle_chunk("Hello")
        handler.handle_chunk(" ")
        handler.handle_chunk("World")
        assert handler.get_complete_response() == "Hello World"

    def test_get_complete_response_empty(self):
        """Test getting complete response when empty."""
        handler = StreamHandler()
        assert handler.get_complete_response() == ""

    def test_clear(self):
        """Test clearing chunks."""
        handler = StreamHandler()
        handler.handle_chunk("Hello")
        handler.handle_chunk("World")
        handler.clear()
        assert handler.chunks == []
        assert handler.get_complete_response() == ""

    def test_finish(self):
        """Test finishing the stream."""
        handler = StreamHandler()
        handler.handle_chunk("Hello")
        handler.handle_chunk("World")
        result = handler.finish()
        assert result == "HelloWorld"
        assert handler.complete is True

    def test_callback_invoked_for_each_chunk(self):
        """Test that callback is invoked for each chunk."""
        callback = Mock()
        handler = StreamHandler(on_chunk=callback)
        handler.handle_chunk("Hello")
        handler.handle_chunk("World")
        assert callback.call_count == 2
        callback.assert_any_call("Hello")
        callback.assert_any_call("World")

    def test_callback_not_invoked_for_empty_chunks(self):
        """Test that callback is not invoked for empty chunks."""
        callback = Mock()
        handler = StreamHandler(on_chunk=callback)
        handler.handle_chunk("")
        handler.handle_chunk("Hello")
        handler.handle_chunk("")
        assert callback.call_count == 1
        callback.assert_called_once_with("Hello")

    def test_callback_exception_does_not_fail_stream(self):
        """Test that callback exception doesn't fail the stream."""
        callback = Mock(side_effect=Exception("Callback error"))
        handler = StreamHandler(on_chunk=callback)
        # Should not raise
        handler.handle_chunk("Hello")
        handler.handle_chunk("World")
        assert handler.chunks == ["Hello", "World"]

    def test_large_chunk_accumulation(self):
        """Test handling large number of chunks."""
        handler = StreamHandler()
        for i in range(1000):
            handler.handle_chunk(str(i))
        assert len(handler.chunks) == 1000
        assert handler.get_complete_response() == "".join(str(i) for i in range(1000))

    def test_long_chunk_size(self):
        """Test handling very long individual chunks."""
        handler = StreamHandler()
        long_text = "x" * 10000
        handler.handle_chunk(long_text)
        assert handler.get_complete_response() == long_text

    def test_unicode_chunks(self):
        """Test handling unicode chunks."""
        handler = StreamHandler()
        handler.handle_chunk("Hello")
        handler.handle_chunk("世界")
        handler.handle_chunk("🌍")
        assert handler.get_complete_response() == "Hello世界🌍"

    def test_newline_chunks(self):
        """Test handling chunks with newlines."""
        handler = StreamHandler()
        handler.handle_chunk("Line 1\n")
        handler.handle_chunk("Line 2\n")
        handler.handle_chunk("Line 3")
        assert handler.get_complete_response() == "Line 1\nLine 2\nLine 3"

    def test_special_characters(self):
        """Test handling special characters."""
        handler = StreamHandler()
        handler.handle_chunk("Tab:\t")
        handler.handle_chunk("Quote:\"")
        handler.handle_chunk("Backslash:\\")
        expected = 'Tab:\tQuote:"Backslash:\\'
        assert handler.get_complete_response() == expected


class TestStreamBuffer:
    """Test StreamBuffer (backwards compatibility alias)."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = StreamBuffer()
        assert buffer._handler is not None

    def test_initialization_with_callback(self):
        """Test buffer initialization with callback."""
        callback = Mock()
        buffer = StreamBuffer(on_chunk=callback)
        assert buffer._handler.on_chunk == callback

    def test_add_chunk(self):
        """Test adding chunks."""
        buffer = StreamBuffer()
        buffer.add_chunk("Hello")
        buffer.add_chunk("World")
        assert buffer.get_complete() == "HelloWorld"

    def test_get_complete(self):
        """Test getting complete content."""
        buffer = StreamBuffer()
        buffer.add_chunk("Hello")
        buffer.add_chunk(" ")
        buffer.add_chunk("World")
        assert buffer.get_complete() == "Hello World"

    def test_clear(self):
        """Test clearing buffer."""
        buffer = StreamBuffer()
        buffer.add_chunk("Hello")
        buffer.clear()
        assert buffer.get_complete() == ""

    def test_callback_integration(self):
        """Test callback through buffer."""
        callback = Mock()
        buffer = StreamBuffer(on_chunk=callback)
        buffer.add_chunk("Hello")
        buffer.add_chunk("World")
        assert callback.call_count == 2


class TestAsyncStreamHandler:
    """Test AsyncStreamHandler class."""

    @pytest.mark.asyncio
    async def test_initialization_without_callback(self):
        """Test async handler initialization without callback."""
        handler = AsyncStreamHandler()
        assert handler.on_chunk is None
        assert handler.chunks == []
        assert handler.complete is False

    @pytest.mark.asyncio
    async def test_initialization_with_callback(self):
        """Test async handler initialization with callback."""
        callback = AsyncMock()
        handler = AsyncStreamHandler(on_chunk=callback)
        assert handler.on_chunk == callback

    @pytest.mark.asyncio
    async def test_handle_single_chunk(self):
        """Test handling a single chunk."""
        handler = AsyncStreamHandler()
        await handler.handle_chunk("Hello")
        assert handler.chunks == ["Hello"]

    @pytest.mark.asyncio
    async def test_handle_multiple_chunks(self):
        """Test handling multiple chunks."""
        handler = AsyncStreamHandler()
        await handler.handle_chunk("Hello")
        await handler.handle_chunk(" ")
        await handler.handle_chunk("World")
        assert handler.chunks == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_handle_empty_chunk_not_added(self):
        """Test that empty chunks are not added."""
        handler = AsyncStreamHandler()
        await handler.handle_chunk("")
        await handler.handle_chunk("Hello")
        await handler.handle_chunk("")
        assert handler.chunks == ["Hello"]

    @pytest.mark.asyncio
    async def test_get_complete_response(self):
        """Test getting complete response."""
        handler = AsyncStreamHandler()
        await handler.handle_chunk("Hello")
        await handler.handle_chunk(" ")
        await handler.handle_chunk("World")
        assert handler.get_complete_response() == "Hello World"

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing chunks."""
        handler = AsyncStreamHandler()
        await handler.handle_chunk("Hello")
        await handler.handle_chunk("World")
        handler.clear()
        assert handler.chunks == []

    @pytest.mark.asyncio
    async def test_finish(self):
        """Test finishing the stream."""
        handler = AsyncStreamHandler()
        await handler.handle_chunk("Hello")
        await handler.handle_chunk("World")
        result = await handler.finish()
        assert result == "HelloWorld"
        assert handler.complete is True

    @pytest.mark.asyncio
    async def test_async_callback_invoked(self):
        """Test that async callback is invoked."""
        callback = AsyncMock()
        handler = AsyncStreamHandler(on_chunk=callback)
        await handler.handle_chunk("Hello")
        await handler.handle_chunk("World")
        assert callback.call_count == 2
        callback.assert_any_call("Hello")
        callback.assert_any_call("World")

    @pytest.mark.asyncio
    async def test_sync_callback_invoked(self):
        """Test that sync callback works with async handler."""
        callback = Mock()
        handler = AsyncStreamHandler(on_chunk=callback)
        await handler.handle_chunk("Hello")
        await handler.handle_chunk("World")
        assert callback.call_count == 2
        callback.assert_any_call("Hello")
        callback.assert_any_call("World")

    @pytest.mark.asyncio
    async def test_callback_not_invoked_for_empty_chunks(self):
        """Test that callback is not invoked for empty chunks."""
        callback = AsyncMock()
        handler = AsyncStreamHandler(on_chunk=callback)
        await handler.handle_chunk("")
        await handler.handle_chunk("Hello")
        await handler.handle_chunk("")
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_fail_stream(self):
        """Test that callback exception doesn't fail the stream."""
        callback = AsyncMock(side_effect=Exception("Callback error"))
        handler = AsyncStreamHandler(on_chunk=callback)
        # Should not raise
        await handler.handle_chunk("Hello")
        await handler.handle_chunk("World")
        assert handler.chunks == ["Hello", "World"]

    @pytest.mark.asyncio
    async def test_large_chunk_accumulation(self):
        """Test handling large number of chunks asynchronously."""
        handler = AsyncStreamHandler()
        for i in range(100):
            await handler.handle_chunk(str(i))
        assert len(handler.chunks) == 100

    @pytest.mark.asyncio
    async def test_unicode_chunks(self):
        """Test handling unicode chunks."""
        handler = AsyncStreamHandler()
        await handler.handle_chunk("Hello")
        await handler.handle_chunk("世界")
        await handler.handle_chunk("🌍")
        assert handler.get_complete_response() == "Hello世界🌍"

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_callbacks(self):
        """Test mixing sync and async operations in handler."""
        sync_callback = Mock()
        handler = AsyncStreamHandler(on_chunk=sync_callback)
        await handler.handle_chunk("Chunk1")
        await handler.handle_chunk("Chunk2")
        assert sync_callback.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_sequential_streams(self):
        """Test handling multiple sequential streams."""
        handler1 = AsyncStreamHandler()
        handler2 = AsyncStreamHandler()

        await handler1.handle_chunk("Stream1")
        await handler2.handle_chunk("Stream2")

        assert handler1.get_complete_response() == "Stream1"
        assert handler2.get_complete_response() == "Stream2"


class TestStreamingEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_stream_handler_state_after_finish(self):
        """Test that handler can still accept chunks after finish."""
        handler = StreamHandler()
        handler.handle_chunk("Hello")
        handler.finish()
        # Can still add chunks after finish
        handler.handle_chunk(" World")
        assert handler.get_complete_response() == "Hello World"

    def test_stream_handler_multiple_clears(self):
        """Test clearing multiple times."""
        handler = StreamHandler()
        handler.handle_chunk("Hello")
        handler.clear()
        handler.clear()
        assert handler.get_complete_response() == ""

    def test_callback_with_none_value(self):
        """Test that handler handles None callback safely."""
        handler = StreamHandler(on_chunk=None)
        handler.handle_chunk("Hello")
        assert handler.get_complete_response() == "Hello"

    @pytest.mark.asyncio
    async def test_async_callback_exception_types(self):
        """Test handling different exception types in async callback."""
        for exc_type in [ValueError, RuntimeError, TypeError, KeyError]:
            callback = AsyncMock(side_effect=exc_type("Test error"))
            handler = AsyncStreamHandler(on_chunk=callback)
            # Should not raise any exception
            await handler.handle_chunk("Test")
            assert handler.chunks == ["Test"]

    def test_whitespace_only_chunks(self):
        """Test handling chunks with only whitespace."""
        handler = StreamHandler()
        handler.handle_chunk("  ")  # spaces
        handler.handle_chunk("\t")  # tab
        handler.handle_chunk("\n")  # newline
        # All should be added since they're not empty
        assert len(handler.chunks) == 3

    def test_stream_handler_repr(self):
        """Test that handler is instantiable and usable."""
        handler = StreamHandler()
        assert hasattr(handler, 'on_chunk')
        assert hasattr(handler, 'chunks')
        assert hasattr(handler, 'complete')

    @pytest.mark.asyncio
    async def test_concurrent_chunk_handling(self):
        """Test handling chunks in rapid succession."""
        handler = AsyncStreamHandler()
        chunks = [f"chunk{i}" for i in range(50)]
        for chunk in chunks:
            await handler.handle_chunk(chunk)
        expected = "".join(chunks)
        assert handler.get_complete_response() == expected
