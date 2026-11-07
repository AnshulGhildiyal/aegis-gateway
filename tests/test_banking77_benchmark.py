"""Tests for Banking77 candidate retrieval helper."""

import asyncio
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aegis.engine.embeddings import FastEmbedder
from aegis.engine.nli_reranker import OnnxNliScorer
from aegis.storage.cache import CacheKey
from benchmarks.evaluate_banking77 import (
    _FROZEN_CONFIGS,
    BankingEvaluationConfig,
    BankingVectorCache,
    _evaluate_config,
)
from benchmarks.evaluate_banking77_loo import (
    _cached_embeddings,
    _calibrate,
    _ensure_nli_scores,
    _load_nli_cache,
    _top_candidates,
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


def test_leave_one_out_candidates_exclude_the_query_row() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])

    candidates = _top_candidates(vectors, vectors, leave_one_out=True)

    assert len(candidates) == 3
    for query_index, row in enumerate(candidates):
        assert len(row) == 2
        assert query_index not in {candidate_index for candidate_index, _ in row}


def test_leave_one_out_calibration_uses_only_train_rows() -> None:
    train = [
        IntentExample("transfer pending", "pending_transfer"),
        IntentExample("transfer not arrived", "transfer_not_received"),
    ]
    rows = [
        [
            {
                "candidate_index": 1,
                "cosine": 0.96,
                "stage0_pass": True,
                "nli_scores": (0.99, 0.99),
                "candidate_intent": train[1].intent,
            }
        ],
        [
            {
                "candidate_index": 0,
                "cosine": 0.96,
                "stage0_pass": True,
                "nli_scores": (0.99, 0.99),
                "candidate_intent": train[0].intent,
            }
        ],
    ]

    selected = _calibrate(train, rows)

    assert len(selected) == 6
    assert all(row["calibration_status"] == "unreachable" for row in selected)


def test_cached_embeddings_are_reused_without_calling_embedder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.evaluate_banking77_loo as loo

    cache_path = tmp_path / "vectors.npz"
    def fake_cache_path(split: str, model: str) -> Path:
        del split, model
        return cache_path

    monkeypatch.setattr(loo, "_vector_cache_path", fake_cache_path)

    class FakeEmbedder:
        model_name = "fake-model"
        calls = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[1.0, 0.0] for _ in texts]

    embedder = FakeEmbedder()
    examples = [IntentExample("query", "intent")]
    first, first_reused = asyncio.run(
        _cached_embeddings(
            examples,
            split="train",
            embedder=cast(FastEmbedder, embedder),
        )
    )
    second, second_reused = asyncio.run(
        _cached_embeddings(
            examples,
            split="train",
            embedder=cast(FastEmbedder, embedder),
        )
    )

    assert first_reused is False
    assert second_reused is True
    assert embedder.calls == 1
    np.testing.assert_array_equal(first, second)


def test_nli_score_cache_reuses_computed_directional_scores(tmp_path: Path) -> None:
    examples = [
        IntentExample("My card was swallowed.", "cash_withdrawal"),
        IntentExample("My card was swallowed by ATM.", "cash_withdrawal"),
    ]
    candidates = [[(1, 0.95)]]

    class FakeScorer:
        calls = 0

        async def entailment_probabilities(
            self,
            pairs: list[tuple[str, str]],
        ) -> list[float]:
            self.calls += 1
            return [0.8, 0.9]

    scorer = FakeScorer()
    cache_path = tmp_path / "scores.jsonl"
    scores = _load_nli_cache(cache_path)
    reused, computed = asyncio.run(
        _ensure_nli_scores(
            [examples[0]],
            examples,
            candidates,
            scorer=cast(OnnxNliScorer, scorer),
            score_cache=scores,
            cache_path=cache_path,
        )
    )
    reused_again, computed_again = asyncio.run(
        _ensure_nli_scores(
            [examples[0]],
            examples,
            candidates,
            scorer=cast(OnnxNliScorer, scorer),
            score_cache=scores,
            cache_path=cache_path,
        )
    )

    assert (reused, computed) == (0, 1)
    assert (reused_again, computed_again) == (1, 0)
    assert scorer.calls == 1
