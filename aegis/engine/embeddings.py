"""Async-friendly text embeddings backed by FastEmbed."""

import asyncio
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from fastembed import TextEmbedding


class EmbeddingModel(Protocol):
    """Minimal interface required from an embedding backend."""

    def embed(self, documents: Iterable[str]) -> Iterable[Iterable[float]]:
        """Return one vector for each input document."""


class FastEmbedder:
    """Run FastEmbed inference in a dedicated thread pool."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        max_workers: int = 1,
        model: EmbeddingModel | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.model_name = model_name
        self._model = model
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="aegis-embedding",
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts without blocking the calling event loop."""
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_sync,
            tuple(texts),
        )

    def _embed_sync(self, texts: tuple[str, ...]) -> list[list[float]]:
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name, lazy_load=True)
        return [
            [float(value) for value in vector] for vector in self._model.embed(texts)
        ]

    async def close(self) -> None:
        """Shut down the dedicated executor without blocking the event loop."""
        await asyncio.to_thread(self._executor.shutdown, wait=True)
