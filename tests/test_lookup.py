"""Tests for three-stage semantic lookup orchestration."""

from collections.abc import Sequence

import pytest

from aegis.engine.lookup import LookupMetric, SemanticLookup
from aegis.engine.nli_reranker import NliDecision
from aegis.storage.cache import CacheEntry, CacheKey


class FakeCache:
    def __init__(self, candidates: list[CacheEntry]) -> None:
        self.candidates = candidates
        self.search_arguments: dict[str, object] | None = None

    async def search(
        self,
        *,
        user_content: str,
        key: CacheKey,
        limit: int = 2,
    ) -> list[CacheEntry]:
        self.search_arguments = {
            "user_content": user_content,
            "key": key,
            "limit": limit,
        }
        return sorted(
            self.candidates,
            key=lambda candidate: candidate.similarity,
            reverse=True,
        )[:limit]


class FakeReranker:
    def __init__(self, accepted: dict[str, bool]) -> None:
        self.accepted = accepted
        self.candidates: list[CacheEntry] = []

    async def rerank(
        self,
        query: str,
        candidates: Sequence[CacheEntry],
    ) -> list[NliDecision]:
        self.candidates = list(candidates)
        return [
            NliDecision(
                candidate=candidate,
                candidate_to_query=(
                    0.95 if self.accepted[candidate.user_content] else 0.2
                ),
                query_to_candidate=(
                    0.9 if self.accepted[candidate.user_content] else 0.3
                ),
                accepted=self.accepted[candidate.user_content],
            )
            for candidate in candidates
        ]


def cache_entry(user_content: str, similarity: float) -> CacheEntry:
    return CacheEntry(
        user_content=user_content,
        response={"answer": user_content},
        created_at=1.0,
        expires_at=2.0,
        last_accessed_at=1.0,
        similarity=similarity,
    )


def key(tenant_id: str = "tenant") -> CacheKey:
    return CacheKey(
        system_prompt_hash="system",
        tenant_id=tenant_id,
        embedding_model_version="embed-v1",
        target_llm="model",
        temperature=0.0,
        max_tokens=128,
    )


async def test_lookup_passes_candidates_through_all_three_stages() -> None:
    first = cache_entry("A cancellation policy covers orders.", 0.91)
    second = cache_entry("A refund policy covers orders.", 0.88)
    cache = FakeCache([second, first])
    reranker = FakeReranker({first.user_content: True, second.user_content: False})
    metrics: list[LookupMetric] = []
    lookup = SemanticLookup(
        cache,
        reranker,
        cosine_threshold=0.8,
        stage0_enabled=True,
        nli_enabled=True,
        metrics_hook=metrics.append,
    )

    result = await lookup.lookup("A cancellation policy covers orders.", key())

    assert result.hit
    assert result.entry is first
    assert cache.search_arguments == {
        "user_content": "A cancellation policy covers orders.",
        "key": key(),
        "limit": 2,
    }
    assert reranker.candidates == [first, second]
    assert [(metric.stage, metric.outcome) for metric in metrics] == [
        ("cosine", "accepted"),
        ("cosine", "accepted"),
        ("stage0", "accepted"),
        ("stage0", "accepted"),
        ("nli", "accepted"),
        ("nli", "rejected"),
    ]


async def test_cosine_rejection_prevents_stage0_and_nli() -> None:
    cache = FakeCache([cache_entry("A policy applies.", 0.4)])
    reranker = FakeReranker({})
    metrics: list[LookupMetric] = []
    lookup = SemanticLookup(
        cache,
        reranker,
        cosine_threshold=0.8,
        stage0_enabled=True,
        metrics_hook=metrics.append,
    )

    result = await lookup.lookup("A policy applies.", key())

    assert not result.hit
    assert reranker.candidates == []
    assert [(metric.stage, metric.outcome) for metric in metrics] == [
        ("cosine", "rejected")
    ]


async def test_stage0_rejection_prevents_nli() -> None:
    cache = FakeCache([cache_entry("Wait 5 days.", 0.9)])
    reranker = FakeReranker({})
    metrics: list[LookupMetric] = []
    lookup = SemanticLookup(
        cache,
        reranker,
        cosine_threshold=0.8,
        stage0_enabled=True,
        nli_enabled=True,
        metrics_hook=metrics.append,
    )

    result = await lookup.lookup("Wait 15 days.", key())

    assert not result.hit
    assert reranker.candidates == []
    assert [(metric.stage, metric.outcome) for metric in metrics] == [
        ("cosine", "accepted"),
        ("stage0", "rejected"),
    ]


async def test_lookup_returns_miss_when_all_nli_candidates_fail() -> None:
    first = cache_entry("A policy applies.", 0.9)
    second = cache_entry("A policy is available.", 0.85)
    cache = FakeCache([first, second])
    reranker = FakeReranker({first.user_content: False, second.user_content: False})
    metrics: list[LookupMetric] = []
    lookup = SemanticLookup(
        cache,
        reranker,
        cosine_threshold=0.8,
        stage0_enabled=True,
        nli_enabled=True,
        metrics_hook=metrics.append,
    )

    result = await lookup.lookup("A policy applies.", key())

    assert not result.hit
    assert result.entry is None
    nli_metrics = [metric for metric in metrics if metric.stage == "nli"]
    assert [metric.outcome for metric in nli_metrics] == ["rejected", "rejected"]
    assert nli_metrics[0].candidate_to_query_entailment == 0.2
    assert nli_metrics[0].query_to_candidate_entailment == 0.3


async def test_lookup_records_empty_vector_search_as_no_candidates() -> None:
    cache = FakeCache([])
    reranker = FakeReranker({})
    metrics: list[LookupMetric] = []
    lookup = SemanticLookup(
        cache,
        reranker,
        cosine_threshold=0.8,
        metrics_hook=metrics.append,
    )

    result = await lookup.lookup("No similar content.", key())

    assert not result.hit
    assert metrics == [
        LookupMetric(
            stage="cosine",
            outcome="no_candidates",
            candidate_rank=None,
        )
    ]
    assert reranker.candidates == []


async def test_only_top_two_candidates_reach_later_stages() -> None:
    candidates = [
        cache_entry("low rank candidate", 0.7),
        cache_entry("highest rank candidate", 0.99),
        cache_entry("second rank candidate", 0.9),
    ]
    cache = FakeCache(candidates)
    reranker = FakeReranker(
        {
            "highest rank candidate": True,
            "second rank candidate": False,
        }
    )
    lookup = SemanticLookup(
        cache,
        reranker,
        cosine_threshold=0.8,
        stage0_enabled=True,
        nli_enabled=True,
    )

    result = await lookup.lookup("highest rank candidate", key())

    assert result.entry is not None
    assert result.entry.user_content == "highest rank candidate"
    assert [item.user_content for item in reranker.candidates] == [
        "highest rank candidate",
        "second rank candidate",
    ]


@pytest.mark.parametrize("threshold", [-1.01, 1.01, float("nan"), float("inf")])
def test_invalid_cosine_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="cosine_threshold"):
        SemanticLookup(FakeCache([]), FakeReranker({}), cosine_threshold=threshold)


async def test_default_lookup_uses_cosine_only() -> None:
    candidate = cache_entry("Wait 5 days.", 0.96)
    reranker = FakeReranker({})
    metrics: list[LookupMetric] = []
    lookup = SemanticLookup(
        FakeCache([candidate]),
        reranker,
        metrics_hook=metrics.append,
    )

    result = await lookup.lookup("Wait 15 days.", key())

    assert result.entry is candidate
    assert reranker.candidates == []
    assert [(metric.stage, metric.outcome) for metric in metrics] == [
        ("cosine", "accepted")
    ]


async def test_cosine_threshold_can_be_overridden_per_tenant() -> None:
    candidate = cache_entry("A policy applies.", 0.9)
    lookup = SemanticLookup(
        FakeCache([candidate]),
        cosine_threshold=0.95,
        tenant_cosine_thresholds={"lower-threshold": 0.85},
    )

    default_tenant = await lookup.lookup("A policy applies.", key())
    configured_tenant = await lookup.lookup(
        "A policy applies.",
        key("lower-threshold"),
    )

    assert not default_tenant.hit
    assert configured_tenant.entry is candidate


def test_nli_cannot_be_enabled_without_a_reranker() -> None:
    with pytest.raises(ValueError, match="reranker is required"):
        SemanticLookup(FakeCache([]), nli_enabled=True)


@pytest.mark.parametrize("threshold", [-1.01, 1.01, float("nan"), float("inf")])
def test_invalid_tenant_cosine_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="cosine_threshold"):
        SemanticLookup(
            FakeCache([]),
            tenant_cosine_thresholds={"tenant": threshold},
        )


async def test_metrics_hook_error_is_not_silently_discarded() -> None:
    def broken_hook(metric: LookupMetric) -> None:
        raise ValueError(metric.stage)

    lookup = SemanticLookup(
        FakeCache([cache_entry("A policy applies.", 0.9)]),
        FakeReranker({}),
        cosine_threshold=0.8,
        metrics_hook=broken_hook,
    )

    with pytest.raises(ValueError, match="cosine"):
        await lookup.lookup("A policy applies.", key())
