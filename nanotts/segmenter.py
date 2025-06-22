from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from dataclasses import dataclass
import re
from typing import Union

import anyio
import tiktoken

from .utils import clean_markdown, preprocess_text

# Tiered separator approach for natural breaking
# Tier 1: Strong punctuation - preferred break points
TIER1_SEPARATORS = re.compile(r"[。！？!?…](?=\s|$)|\.(?=\s+[A-Z])|\.(?=\s*$)|\n+")

# Tier 2: Soft punctuation - fallback when approaching max tokens
TIER2_SEPARATORS = re.compile(r",\s+")

# Combined pattern for backward compatibility
SENTENCE_ENDINGS = re.compile(r"[。！？!?…](?=\s|$)|\.(?=\s+[A-Z])|\.(?=\s*$)|,\s+|\n+")


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
        min_tokens: int = 10,
        max_tokens: int = 50,
        token: StreamToken,
    ):
        self._send_stream = send_stream
        self._pre_hook = pre_hook
        self._timeout_ms = timeout_ms / 1000.0
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._token = token

        # Initialize tiktoken encoder (GPT-4 encoding)
        self._encoder = tiktoken.get_encoding("cl100k_base")

        # Segmentation state
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
        """Process string with timeout handling - now using token-based logic."""
        await self._process_string(text)

    async def _process_string(self, text: str) -> None:
        """Process string with token-based segmentation."""
        if not text:  # Only skip if completely empty, not just whitespace
            return

        # Check cancellation before processing
        if self._token.cancelled():
            return

        # Preprocess text (clean markdown, normalize) - preserves whitespace
        processed_text = preprocess_text(text)
        if processed_text is None:  # Only skip if preprocessing returns None
            return

        self._buf += processed_text

        # Check for segmentation opportunities
        await self._check_and_segment()

    async def _check_and_segment(self) -> None:
        """Check if buffer should be segmented using tiered separator approach."""
        if not self._buf.strip():
            return

        # Get current token count
        current_tokens = len(self._encoder.encode(self._buf))

        # Force segmentation if we hit max tokens
        if current_tokens >= self._max_tokens:
            await self._emit_with_token_boundary()
            return

        # Tiered approach: try Tier 1 first if we have minimum tokens
        if current_tokens >= self._min_tokens and await self._try_segment_at_tier(TIER1_SEPARATORS):
            return

        # If approaching max tokens but Tier 1 failed, try Tier 2
        if current_tokens >= self._max_tokens * 0.8 and await self._try_segment_at_tier(TIER2_SEPARATORS):
            return

    async def _try_segment_at_tier(self, pattern: re.Pattern) -> bool:
        """Try to segment using the given separator pattern. Returns True if segmented."""
        break_point = self._find_break_point_with_pattern(pattern)

        if break_point > 0:
            # Found a good break point - emit up to here
            segment_text = self._buf[:break_point]
            remaining = self._buf[break_point:]

            self._buf = segment_text
            await self._emit()

            # Keep remaining text in buffer
            self._buf = remaining

            # Only recurse if remaining text has substantial content
            if (
                remaining.strip()
                and len(self._encoder.encode(remaining)) >= self._min_tokens
            ):
                await self._check_and_segment()  # Use main logic for recursion
            return True

        return False  # No suitable break point found

    def _find_break_point_with_pattern(self, pattern: re.Pattern) -> int:
        """Find the first valid break point with sufficient tokens using given pattern."""
        for match in pattern.finditer(self._buf):
            end_pos = match.end()
            segment_text = self._buf[:end_pos]
            token_count = len(self._encoder.encode(segment_text))

            if token_count >= self._min_tokens:
                return end_pos

        return 0  # No suitable break point found

    async def _emit_with_token_boundary(self) -> None:
        """Emit segment respecting token boundaries when hitting max tokens."""
        if not self._buf.strip():
            return

        tokens = self._encoder.encode(self._buf)

        # If we're not over limit, just emit normally
        if len(tokens) <= self._max_tokens:
            await self._emit()
            return

        # Find the best break point within token limit
        best_break = self._find_token_break_point(tokens)

        # Split at the break point
        text_to_emit = self._encoder.decode(tokens[:best_break])
        remaining_tokens = tokens[best_break:]

        if text_to_emit.strip():
            self._buf = text_to_emit
            await self._emit(token_boundary=True)

        # Keep remaining text in buffer
        self._buf = self._encoder.decode(remaining_tokens)

    def _find_token_break_point(self, tokens: list) -> int:
        """Find optimal token break point within limits."""
        max_search = min(len(tokens), self._max_tokens)

        # Look backwards from max_tokens to find any separator (tier 1 or 2)
        for i in range(max_search, self._min_tokens, -1):
            partial_text = self._encoder.decode(tokens[:i])
            if TIER1_SEPARATORS.search(partial_text) or TIER2_SEPARATORS.search(
                partial_text
            ):
                return i

        # If no sentence break found, look for word boundaries
        for i in range(self._max_tokens, self._min_tokens, -1):
            if i >= len(tokens):
                continue
            partial_text = self._encoder.decode(tokens[:i])
            if partial_text.endswith((" ", "\n", "\t")):
                return i

        return min(self._max_tokens, len(tokens))

    async def _emit(
        self, *, smart_break: bool = False, token_boundary: bool = False
    ) -> None:
        """Unified emit method with optional smart breaking and token boundary handling."""
        if not self._buf:
            return

        text = self._buf
        if not text.strip():  # Only skip if the text is purely whitespace
            self._buf = ""
            return

        # Handle smart breaking if requested
        if smart_break and not token_boundary:
            break_point = self._find_smart_break_point()
            if break_point < len(text):
                text = text[:break_point]
                self._buf = self._buf[break_point:]  # Keep remaining
            else:
                self._buf = ""  # Using full buffer
        else:
            self._buf = ""  # Clear buffer for normal emit

        # Apply markdown cleaning to complete segment text
        if text.strip():  # Only clean non-whitespace text
            text = clean_markdown(text)

        if self._pre_hook:
            text = await self._pre_hook(text)

        if text:  # Only emit if there's content after processing
            segment = Segment(self._segment_id, text)
            self._segment_id += 1
            await self._send_stream.send(segment)

    def _find_smart_break_point(self) -> int:
        """Find optimal word boundary break point within last 20 characters."""
        if len(self._buf) <= 20:
            return len(self._buf)

        search_start = max(0, len(self._buf) - 20)

        # Look backwards for good break points
        for i in range(len(self._buf) - 1, search_start - 1, -1):
            char = self._buf[i]
            if char in " ,;:" or char in "-—–":
                return i + 1

        return len(self._buf)  # No good break found

    async def flush(self) -> None:
        await self._emit()
