from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from dataclasses import dataclass
import re
from typing import Union

import anyio

STRONG_PUNCT = re.compile(r"[。！？!?…\.,，]")


@dataclass
class Segment:
    id: int
    text: str


class StreamToken:
    def __init__(self):
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def cancelled(self) -> bool:
        return self._cancelled

    async def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise anyio.get_cancelled_exc_class()()


class Segmenter:
    def __init__(
        self,
        send_stream: anyio.abc.ObjectSendStream,
        *,
        pre_hook: Callable[[str], Awaitable[str]] | None = None,
        timeout_ms: int = 800,
        max_len: int = 120,
        token: StreamToken,
    ):
        self._send_stream = send_stream
        self._pre_hook = pre_hook
        self._timeout_ms = timeout_ms / 1000.0
        self._max_len = max_len
        self._token = token
        self._buf = ""
        self._segment_id = 0

    async def feed(
        self, text_or_iter: Union[str, Iterable[str], AsyncIterable[str]]
    ) -> None:
        if isinstance(text_or_iter, str):
            await self._process_string(text_or_iter)
        elif hasattr(text_or_iter, "__aiter__"):
            await self._process_async_iter_with_timeout(text_or_iter)
        else:
            for chunk in text_or_iter:
                await self._process_string_with_timeout(chunk)
                if self._token.cancelled():
                    return

        await self.flush()

    async def _process_async_iter_with_timeout(self, text_iter) -> None:
        """Process async iterator with proper timeout logic."""
        # Use iter() for Python 3.9 compatibility
        async_iter = text_iter.__aiter__()

        while True:
            if self._token.cancelled():
                return

            try:
                # Wait for next chunk with timeout
                with anyio.move_on_after(self._timeout_ms):
                    chunk = await async_iter.__anext__()
                    # Process the chunk immediately
                    await self._process_string_with_timeout(chunk)
                    continue

                # Timeout occurred - emit current buffer if it has content
                if self._buf:
                    await self._emit()
                    continue
                else:
                    # No content to emit and timeout - we're done
                    break

            except StopAsyncIteration:
                # Iterator exhausted
                break

    async def _process_string_with_timeout(self, text: str) -> None:
        for char in text:
            if self._token.cancelled():
                return

            self._buf += char

            # Priority 1: Strong punctuation - always emit immediately
            if STRONG_PUNCT.search(char):
                await self._emit()
            # Priority 2: Max length - but try to find nearby punctuation first
            elif len(self._buf) >= self._max_len:
                await self._emit_with_smart_break()

    async def _process_string(self, text: str) -> None:
        for char in text:
            if self._token.cancelled():
                return

            self._buf += char

            # Priority 1: Strong punctuation - always emit immediately
            if STRONG_PUNCT.search(char):
                await self._emit()
            # Priority 2: Max length - but try to find nearby punctuation first
            elif len(self._buf) >= self._max_len:
                await self._emit_with_smart_break()

    async def _emit_with_smart_break(self) -> None:
        """Emit text with smart breaking - prefer word boundaries over character splits."""
        if not self._buf:
            return

        # If we're at max length, try to find a good break point
        # Look for spaces, commas, or other soft punctuation within the last 20 chars
        break_point = len(self._buf)
        search_start = max(0, len(self._buf) - 20)

        # Look backwards for good break points
        for i in range(len(self._buf) - 1, search_start - 1, -1):
            char = self._buf[i]
            # Prefer word boundaries (spaces, commas, semicolons)
            if char in " ,;:" or char in "-—–":
                break_point = i + 1
                break

        # If we found a good break point, split there
        if break_point < len(self._buf):
            text_to_emit = self._buf[:break_point]
            remaining = self._buf[break_point:].lstrip()  # Remove leading spaces

            # Emit the first part
            if text_to_emit.strip():
                if self._pre_hook:
                    text_to_emit = await self._pre_hook(text_to_emit.strip())
                else:
                    text_to_emit = text_to_emit.strip()

                if text_to_emit:
                    segment = Segment(self._segment_id, text_to_emit)
                    self._segment_id += 1
                    await self._send_stream.send(segment)

            # Keep the remaining part in buffer
            self._buf = remaining
        else:
            # No good break point found, just emit normally
            await self._emit()

    async def _emit(self) -> None:
        if not self._buf:
            return

        text = self._buf.strip()
        if not text:
            self._buf = ""
            return

        if self._pre_hook:
            text = await self._pre_hook(text)

        segment = Segment(self._segment_id, text)
        self._segment_id += 1
        self._buf = ""

        await self._send_stream.send(segment)

    async def flush(self) -> None:
        await self._emit()
