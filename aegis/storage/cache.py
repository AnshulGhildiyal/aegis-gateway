"""Embedded Qdrant storage for semantically retrieved completion candidates."""

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException

_EMBEDDING_OPERATION_ERRORS = (
    OSError,
    HfHubHTTPError,
)
_QDRANT_OPERATION_ERRORS = (OSError, RuntimeError, ApiException)


class CacheOperationalError(Exception):
    """Raised when an embedding or Qdrant backend operation fails."""


class AsyncEmbedder(Protocol):
    """Embedding interface consumed by the semantic cache."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for each input text."""


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Exact-match dimensions that isolate otherwise similar requests."""

    system_prompt_hash: str
    tenant_id: str
    embedding_model_version: str
    target_llm: str
    temperature: float
    max_tokens: int | None
    tools: list[dict[str, object]] | None = None
    response_format: dict[str, object] | None = None

    def payload_values(self) -> dict[str, str]:
        """Return canonical string values suitable for exact Qdrant filters."""
        values: dict[str, object] = {
            "system_prompt_hash": self.system_prompt_hash,
            "tenant_id": self.tenant_id,
            "embedding_model_version": self.embedding_model_version,
            "target_llm": self.target_llm,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "response_format": self.response_format,
        }
        return {
            f"key_{name}": json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for name, value in values.items()
        }


class CacheEntry(BaseModel):
    """Cached completion and its TTL/LRU metadata."""

    user_content: str
    response: dict[str, object]
    created_at: float
    expires_at: float
    last_accessed_at: float
    similarity: float


class _StoredPayload(BaseModel):
    """Validated shape of payloads read back from Qdrant."""

    user_content: str
    response: dict[str, object]
    created_at: float
    expires_at: float
    last_accessed_at: float


class SemanticCache:
    """Store and retrieve vector candidates using an embedded Qdrant collection."""

    def __init__(
        self,
        embedder: AsyncEmbedder,
        *,
        path: str = ":memory:",
        collection_name: str = "aegis_cache",
        vector_size: int = 384,
        ttl_seconds: float = 3600,
        max_entries: int = 10_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if vector_size < 1:
            raise ValueError("vector_size must be at least 1")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and greater than 0")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")

        self._embedder = embedder
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._lock = asyncio.Lock()
        self._client = QdrantClient(path=path)
        self._collection_name = collection_name
        try:
            if not self._client.collection_exists(collection_name):
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
        except BaseException as initialization_error:
            try:
                self._client.close()
            except BaseException as cleanup_error:
                initialization_error.add_note(
                    f"Qdrant cleanup also failed: {cleanup_error!r}"
                )
            raise
        self._vector_size = vector_size

    async def put(
        self,
        *,
        user_content: str,
        response: dict[str, object],
        key: CacheKey,
    ) -> None:
        """Embed and store a completion, applying TTL cleanup and LRU eviction."""
        try:
            vectors = await self._embedder.embed([user_content])
        except _EMBEDDING_OPERATION_ERRORS as error:
            raise CacheOperationalError("Cache embedding operation failed.") from error
        vector = self._validate_vectors(vectors, expected_count=1)[0]
        now = self._clock()
        payload = {
            **key.payload_values(),
            "user_content": user_content,
            "response": response,
            "created_at": now,
            "expires_at": now + self._ttl_seconds,
            "last_accessed_at": now,
        }
        async with self._lock:
            try:
                await asyncio.to_thread(self._put_sync, vector, payload, now)
            except _QDRANT_OPERATION_ERRORS as error:
                raise CacheOperationalError(
                    "Cache storage operation failed."
                ) from error

    async def search(
        self,
        *,
        user_content: str,
        key: CacheKey,
        limit: int = 2,
    ) -> list[CacheEntry]:
        """Return unexpired, exact-key candidates ordered by cosine similarity."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        try:
            vectors = await self._embedder.embed([user_content])
        except _EMBEDDING_OPERATION_ERRORS as error:
            raise CacheOperationalError("Cache embedding operation failed.") from error
        vector = self._validate_vectors(vectors, expected_count=1)[0]
        now = self._clock()
        async with self._lock:
            try:
                return await asyncio.to_thread(
                    self._search_sync,
                    vector,
                    key,
                    limit,
                    now,
                )
            except _QDRANT_OPERATION_ERRORS as error:
                raise CacheOperationalError(
                    "Cache search operation failed."
                ) from error

    def close(self) -> None:
        """Close the embedded Qdrant client."""
        self._client.close()

    def _put_sync(
        self,
        vector: list[float],
        payload: dict[str, object],
        now: float,
    ) -> None:
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload=payload,
                )
            ],
            wait=True,
        )
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="expires_at",
                            range=models.Range(lte=now),
                        )
                    ]
                )
            ),
            wait=True,
        )
        self._evict_over_capacity()

    def _search_sync(
        self,
        vector: list[float],
        key: CacheKey,
        limit: int,
        now: float,
    ) -> list[CacheEntry]:
        query_filter = models.Filter(
            must=[
                *self._key_conditions(key),
                models.FieldCondition(
                    key="expires_at",
                    range=models.Range(gt=now),
                ),
            ]
        )
        result = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        entries: list[CacheEntry] = []
        for point in result.points:
            if point.payload is None:
                continue
            payload = _StoredPayload.model_validate(point.payload)
            self._client.set_payload(
                collection_name=self._collection_name,
                payload={"last_accessed_at": now},
                points=[point.id],
                wait=True,
            )
            entries.append(
                CacheEntry(
                    user_content=payload.user_content,
                    response=payload.response,
                    created_at=payload.created_at,
                    expires_at=payload.expires_at,
                    last_accessed_at=now,
                    similarity=point.score,
                )
            )
        return entries

    def _evict_over_capacity(self) -> None:
        count = self._client.count(
            collection_name=self._collection_name,
            exact=True,
        ).count
        excess = count - self._max_entries
        if excess <= 0:
            return
        points, _ = self._client.scroll(
            collection_name=self._collection_name,
            limit=count,
            with_payload=True,
            with_vectors=False,
        )
        oldest = sorted(
            points,
            key=lambda point: (
                (_StoredPayload.model_validate(point.payload).last_accessed_at)
                if point.payload is not None
                else float("-inf")
            ),
        )[:excess]
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[point.id for point in oldest]),
            wait=True,
        )

    def _key_conditions(self, key: CacheKey) -> list[models.FieldCondition]:
        return [
            models.FieldCondition(
                key=field,
                match=models.MatchValue(value=value),
            )
            for field, value in key.payload_values().items()
        ]

    def _validate_vectors(
        self,
        vectors: list[list[float]],
        *,
        expected_count: int,
    ) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise ValueError(
                f"embedder returned {len(vectors)} vectors; expected {expected_count}"
            )
        for vector in vectors:
            if len(vector) != self._vector_size:
                raise ValueError(
                    f"embedding dimension {len(vector)} does not match "
                    f"configured vector_size {self._vector_size}"
                )
        return vectors
