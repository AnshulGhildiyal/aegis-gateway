"""Tests for Banking77 candidate retrieval helper."""

import asyncio
from typing import cast

import numpy as np
import pytest

from aegis.engine.nli_reranker import OnnxNliScorer
from aegis.storage.cache import CacheKey
from benchmarks.evaluate_banking77 import (
    _FROZEN_CONFIGS,
    BankingEvaluationConfig,
    BankingVectorCache,
    _evaluate_config,
)
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


def test_frozen_configs_match_requested_thresholds() -> None:
    assert [
        (config.cosine_threshold, config.nli_threshold) for config in _FROZEN_CONFIGS
    ] == [
        (0.98, None),
        (0.95, 0.97),
        (0.85, 0.23),
        (0.92, None),
    ]


def test_evaluate_config_reports_false_intent_hits_and_nli_call_rate() -> None:
    train = [
        IntentExample("Replace the card today.", "cash_withdrawal"),
        IntentExample("Replace the card now.", "cash_withdrawal"),
    ]
    queries = [
        IntentExample("Please replace the card.", "cash_deposit"),
        IntentExample("Replace a card.", "cash_deposit"),
    ]
    vectors = np.asarray([[1.0, 0.0], [0.99, 0.01]])
    query_vectors = np.asarray([[1.0, 0.0], [0.99, 0.01]])
    cache = BankingVectorCache(train, vectors)
    key = CacheKey(
        system_prompt_hash="test",
        tenant_id="test",
        embedding_model_version="test",
        target_llm="test",
        temperature=0.0,
        max_tokens=None,
    )

    class FakeScorer:
        async def entailment_probabilities(
            self,
            pairs: list[tuple[str, str]],
        ) -> list[float]:
            return [0.99 for _ in pairs]

    config = BankingEvaluationConfig("test-nli", 0.8, 0.9)
    summary, confusions = asyncio.run(
        _evaluate_config(
            config,
            queries=queries,
            query_vectors=query_vectors,
            cache=cache,
            key=key,
            nli_scorer=cast(OnnxNliScorer, FakeScorer()),
        )
    )

    assert summary["hits"] == 2
    assert summary["false_positives"] == 2
    assert summary["true_intent_hits"] == 0
    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0
    assert summary["nli_queries"] == 2
    assert summary["nli_call_rate"] == 1.0
    assert summary["nli_candidate_pairs_scored"] == 4
    assert confusions == [
        {
            "config": "test-nli",
            "rank": 1,
            "query_intent": "cash_deposit",
            "retrieved_intent": "cash_withdrawal",
            "false_positive_count": 2,
        }
    ]


def test_no_nli_config_does_not_score_or_report_nli_calls() -> None:
    examples = [IntentExample("My card was swallowed.", "cash_withdrawal")]
    vectors = np.asarray([[1.0, 0.0]])
    cache = BankingVectorCache(examples, vectors)
    key = CacheKey(
        system_prompt_hash="test",
        tenant_id="test",
        embedding_model_version="test",
        target_llm="test",
        temperature=0.0,
        max_tokens=None,
    )

    class FakeScorer:
        async def entailment_probabilities(
            self,
            pairs: list[tuple[str, str]],
        ) -> list[float]:
            raise AssertionError("NLI must not be called for a Stage-0-only config")

    summary, _ = asyncio.run(
        _evaluate_config(
            BankingEvaluationConfig("test-cosine-stage0", 0.8, None),
            queries=examples,
            query_vectors=vectors,
            cache=cache,
            key=key,
            nli_scorer=cast(OnnxNliScorer, FakeScorer()),
        )
    )

    assert summary["hits"] == 1
    assert summary["nli_queries"] == 0
    assert summary["nli_call_rate"] == 0.0
