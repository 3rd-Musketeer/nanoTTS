from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable
from typing import Union

import anyio

from .audio_data import AudioChunk, AudioSpec, AudioTranscoder
from .engine import Engine
from .model import manager
from .segmenter import Segment, Segmenter, StreamToken


class NanoTTS:
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        model: str | None = None,
        output_spec: AudioSpec | None = None,
        timeout_ms: int = 800,
        min_tokens: int = 10,
        max_tokens: int = 50,
        pre_hook: Callable[[str], Awaitable[str]] | None = None,
        **engine_kwargs,
    ):
        if engine is None and model is None:
            model = "dummy"

        if engine is not None and model is not None:
            raise ValueError("Cannot specify both engine and model")

        self._engine = engine
        self._model = model
        self._engine_kwargs = engine_kwargs
        self._output_spec = output_spec or AudioSpec("pcm", 16000, 1, 16)
        self._timeout_ms = timeout_ms
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._pre_hook = pre_hook
        self._token: StreamToken | None = None

    def cancel(self) -> None:
        if self._token:
            self._token.cancel()

    async def stream(
        self, text_or_iter: Union[str, Iterable[str], AsyncIterable[str]]
    ) -> AsyncIterator[tuple[AudioChunk, str]]:
        self._token = StreamToken()

        if self._engine is None:
            self._engine = await manager.get(self._model, **self._engine_kwargs)

        segment_send, segment_recv = anyio.create_memory_object_stream(
            max_buffer_size=64
        )
        audio_send, audio_recv = anyio.create_memory_object_stream(max_buffer_size=64)

        async with anyio.create_task_group() as tg:
            tg.start_soon(self._segment_producer, text_or_iter, segment_send)
            tg.start_soon(self._tts_worker, segment_recv, audio_send)

            async for audio_chunk, text in self._reorder_consumer(audio_recv):
                if self._token.cancelled():
                    return
                yield audio_chunk, text

    async def _segment_producer(
        self,
        text_or_iter: Union[str, Iterable[str], AsyncIterable[str]],
        send_stream: anyio.abc.ObjectSendStream[Segment],
    ) -> None:
        try:
            segmenter = Segmenter(
                send_stream,
                pre_hook=self._pre_hook,
                timeout_ms=self._timeout_ms,
                min_tokens=self._min_tokens,
                max_tokens=self._max_tokens,
                token=self._token,
            )
            await segmenter.feed(text_or_iter)
        except anyio.get_cancelled_exc_class():
            pass
        finally:
            await send_stream.aclose()

    async def _tts_worker(
        self,
        recv_stream: anyio.abc.ObjectReceiveStream[Segment],
        send_stream: anyio.abc.ObjectSendStream[tuple[int, AudioChunk, str]],
    ) -> None:
        try:
            async for segment in recv_stream:
                if self._token.cancelled():
                    break

                try:
                    raw_chunk = await self._engine.synth(
                        segment.text, target=self._output_spec
                    )
                    final_chunk = await AudioTranscoder.convert(
                        raw_chunk, self._output_spec
                    )
                    await send_stream.send((segment.id, final_chunk, segment.text))
                except Exception:
                    # Skip failed segments
                    continue
        except anyio.get_cancelled_exc_class():
            pass
        finally:
            await send_stream.aclose()

    async def _reorder_consumer(
        self, recv_stream: anyio.abc.ObjectReceiveStream[tuple[int, AudioChunk, str]]
    ) -> AsyncIterator[tuple[AudioChunk, str]]:
        expected_id = 0
        pending: dict[int, tuple[AudioChunk, str]] = {}

        try:
            async for segment_id, audio_chunk, text in recv_stream:
                if self._token.cancelled():
                    return

                if segment_id == expected_id:
                    yield audio_chunk, text
                    expected_id += 1

                    # Check if we can yield any pending segments
                    while expected_id in pending:
                        audio_chunk, text = pending.pop(expected_id)
                        yield audio_chunk, text
                        expected_id += 1
                else:
                    pending[segment_id] = (audio_chunk, text)
        except anyio.get_cancelled_exc_class():
            pass
