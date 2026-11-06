"""Tests for Banking77 candidate retrieval helper."""

import numpy as np
import pytest

from aegis.storage.cache import CacheKey
from benchmarks.evaluate_banking77 import BankingVectorCache
from benchmarks.evaluation import IntentExample


async def test_vector_cache_returns_nearest_labeled_candidates() -> None:
    examples = [
        IntentExample("one", "intent_one"),
        IntentExample("two", "intent_two"),
        IntentExample("three", "intent_three"),
    ]
    cache = BankingVectorCache(
        examples,
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
    )
    cache.query_vector = np.asarray([0.9, 0.1])
    key = CacheKey(
        system_prompt_hash="test",
        tenant_id="test",
        embedding_model_version="test",
        target_llm="test",
        temperature=0.0,
        max_tokens=None,
    )

    entries = await cache.search(user_content="query", key=key)

    assert [entry.user_content for entry in entries] == ["one", "two"]
    assert [entry.response["intent"] for entry in entries] == [
        "intent_one",
        "intent_two",
    ]


def test_vector_cache_rejects_mismatched_training_vectors() -> None:
    with pytest.raises(ValueError, match="one vector"):
        BankingVectorCache(
            [IntentExample("only", "intent")],
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        )
