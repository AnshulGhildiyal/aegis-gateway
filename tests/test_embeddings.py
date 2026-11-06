"""Tests for FastEmbed's asynchronous wrapper."""

import threading
from collections.abc import Iterable

import pytest

from aegis.engine.embeddings import FastEmbedder


class FakeModel:
    def __init__(self) -> None:
        self.thread_id: int | None = None

    def embed(self, documents: Iterable[str]) -> Iterable[Iterable[float]]:
        self.thread_id = threading.get_ident()
        return ([float(len(document)), 1.0] for document in documents)


async def test_embedding_runs_in_executor_and_returns_vectors() -> None:
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    event_loop_thread = threading.get_ident()
    try:
        result = await embedder.embed(["hello", "world!"])
    finally:
        await embedder.close()

    assert result == [[5.0, 1.0], [6.0, 1.0]]
    assert model.thread_id is not None
    assert model.thread_id != event_loop_thread


async def test_empty_input_does_not_invoke_model() -> None:
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    try:
        assert await embedder.embed([]) == []
    finally:
        await embedder.close()

    assert model.thread_id is None


async def test_invalid_worker_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        FastEmbedder(max_workers=0)
