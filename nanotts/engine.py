from __future__ import annotations

from typing import Callable, Protocol

import anyio

from .audio_data import AudioChunk, AudioSpec


class Engine(Protocol):
    async def synth(
        self, text: str, *, target: AudioSpec | None = None
    ) -> AudioChunk: ...


class CallableEngine(Engine):
    def __init__(self, fn: Callable[[str], bytes], *, output_spec: AudioSpec):
        self._fn = fn
        self._output_spec = output_spec

    async def synth(self, text: str, *, target: AudioSpec | None = None) -> AudioChunk:
        audio_bytes = await anyio.to_thread.run_sync(self._fn, text)
        return AudioChunk(audio_bytes, self._output_spec)
