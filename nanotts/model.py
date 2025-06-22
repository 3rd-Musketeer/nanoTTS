from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .engine import Engine

CACHE_DIR = Path.home() / ".cache" / "nano_tts"


@dataclass(slots=True)
class EngineFactory:
    build: Callable[..., Awaitable[Engine]]
    doc: str


class ModelManager:
    def __init__(self):
        self._factories: dict[str, EngineFactory] = {}
        self._download_locks: dict[str, asyncio.Lock] = {}

    def register(
        self, name: str, build: Callable[..., Awaitable[Engine]], doc: str = ""
    ) -> None:
        self._factories[name] = EngineFactory(build, doc)

    async def get(self, name: str = "dummy", **engine_kwargs) -> Engine:
        if name not in self._factories:
            raise ValueError(f"Unknown model: {name}")

        factory = self._factories[name]
        return await factory.build(**engine_kwargs)

    def list_models(self) -> dict[str, str]:
        return {name: factory.doc for name, factory in self._factories.items()}

    async def download_model(self, name: str, url: str) -> None:
        if name not in self._download_locks:
            self._download_locks[name] = asyncio.Lock()

        async with self._download_locks[name]:
            cache_path = CACHE_DIR / name
            if cache_path.exists():
                return

            cache_path.parent.mkdir(parents=True, exist_ok=True)

            raise NotImplementedError(
                "Model downloading not implemented - use external tools"
            )


manager = ModelManager()
