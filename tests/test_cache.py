"""Tests for embedded Qdrant semantic cache storage."""

from collections.abc import Iterable

import pytest

from aegis.storage.cache import CacheKey, SemanticCache


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vector_for(text) for text in texts]

    @staticmethod
    def vector_for(text: str) -> list[float]:
        vectors = {
            "first": [1.0, 0.0, 0.0],
            "second": [0.0, 1.0, 0.0],
            "query": [1.0, 0.0, 0.0],
        }
        return vectors.get(text, [0.0, 0.0, 1.0])


def cache_key(
    *,
    system_prompt_hash: str = "system-hash",
    tenant_id: str = "tenant-a",
    embedding_model_version: str = "embed-v1",
    target_llm: str = "model-a",
    temperature: float = 0.0,
    max_tokens: int | None = 128,
    tools: list[dict[str, object]] | None = None,
    response_format: dict[str, object] | None = None,
) -> CacheKey:
    return CacheKey(
        system_prompt_hash=system_prompt_hash,
        tenant_id=tenant_id,
        embedding_model_version=embedding_model_version,
        target_llm=target_llm,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        response_format=response_format,
    )


@pytest.fixture
def cache() -> Iterable[SemanticCache]:
    instance = SemanticCache(FakeEmbedder(), vector_size=3)
    try:
        yield instance
    finally:
        instance.close()


async def test_put_and_search_return_exact_key_candidate(
    cache: SemanticCache,
) -> None:
    await cache.put(
        user_content="first",
        response={"answer": "stored"},
        key=cache_key(),
    )

    entries = await cache.search(user_content="query", key=cache_key())

    assert len(entries) == 1
    assert entries[0].response == {"answer": "stored"}
    assert entries[0].similarity == pytest.approx(1.0)
    assert entries[0].expires_at > entries[0].created_at


async def test_search_filters_every_strict_cache_key_field(
    cache: SemanticCache,
) -> None:
    await cache.put(
        user_content="first",
        response={"answer": "tenant A"},
        key=cache_key(),
    )

    mismatched_keys = [
        cache_key(system_prompt_hash="other-system-hash"),
        cache_key(tenant_id="tenant-b"),
        cache_key(embedding_model_version="embed-v2"),
        cache_key(target_llm="model-b"),
        cache_key(temperature=0.1),
        cache_key(max_tokens=256),
        cache_key(tools=[{"type": "function"}]),
        cache_key(response_format={"type": "json_object"}),
    ]
    for key in mismatched_keys:
        assert await cache.search(user_content="query", key=key) == []


async def test_expired_entries_are_not_returned() -> None:
    now = [100.0]
    instance = SemanticCache(
        FakeEmbedder(),
        vector_size=3,
        ttl_seconds=10,
        clock=lambda: now[0],
    )
    try:
        await instance.put(
            user_content="first",
            response={"answer": "expired"},
            key=cache_key(),
        )
        now[0] += 10

        entries = await instance.search(user_content="query", key=cache_key())

        assert entries == []
    finally:
        instance.close()


async def test_capacity_evicts_least_recently_used_entry() -> None:
    now = [100.0]
    instance = SemanticCache(
        FakeEmbedder(),
        vector_size=3,
        max_entries=2,
        clock=lambda: now[0],
    )
    try:
        await instance.put(
            user_content="first",
            response={"answer": "first"},
            key=cache_key(),
        )
        now[0] += 1
        await instance.put(
            user_content="second",
            response={"answer": "second"},
            key=cache_key(),
        )
        now[0] += 1
        assert await instance.search(user_content="first", key=cache_key(), limit=1)
        now[0] += 1
        await instance.put(
            user_content="third",
            response={"answer": "third"},
            key=cache_key(),
        )

        remaining = await instance.search(user_content="second", key=cache_key())
        assert all(entry.user_content != "second" for entry in remaining)
        assert len(await instance.search(user_content="query", key=cache_key())) == 2
    finally:
        instance.close()


async def test_wrong_embedding_dimension_is_rejected() -> None:
    instance = SemanticCache(FakeEmbedder(), vector_size=4)
    try:
        with pytest.raises(ValueError, match="embedding dimension"):
            await instance.put(
                user_content="first",
                response={"answer": "stored"},
                key=cache_key(),
            )
    finally:
        instance.close()


def test_invalid_cache_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        SemanticCache(FakeEmbedder(), max_entries=0)
