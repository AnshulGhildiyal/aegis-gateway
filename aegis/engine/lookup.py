"""Configurable semantic cache lookup orchestration."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from aegis.engine.nli_reranker import NliDecision, NliReranker
from aegis.engine.regex_guard import stage0_guard
from aegis.storage.cache import CacheEntry, CacheKey, SemanticCache

_MAX_CANDIDATES = 2
_DEFAULT_COSINE_THRESHOLD = 0.95


class CandidateCache(Protocol):
    """Cache API required by the semantic lookup pipeline."""

    async def search(
        self,
        *,
        user_content: str,
        key: CacheKey,
        limit: int = 2,
    ) -> list[CacheEntry]:
        """Return vector-retrieved candidates for a strict cache key."""


class CandidateReranker(Protocol):
    """Bidirectional NLI API required by the lookup pipeline."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[CacheEntry],
    ) -> list[NliDecision]:
        """Return entailment decisions for the supplied candidates."""


LookupStage = Literal["cosine", "stage0", "nli"]
LookupOutcome = Literal["accepted", "rejected", "no_candidates"]


@dataclass(frozen=True, slots=True)
class LookupMetric:
    """One per-candidate lookup stage event, without user text."""

    stage: LookupStage
    outcome: LookupOutcome
    candidate_rank: int | None
    cosine_similarity: float | None = None
    candidate_to_query_entailment: float | None = None
    query_to_candidate_entailment: float | None = None


MetricHook = Callable[[LookupMetric], None]


@dataclass(frozen=True, slots=True)
class LookupResult:
    """Result of one semantic cache lookup."""

    entry: CacheEntry | None

    @property
    def hit(self) -> bool:
        """Whether the enabled lookup checks accepted a cache entry."""
        return self.entry is not None


class SemanticLookup:
    """Apply cosine matching with optional Stage-0 and bidirectional NLI checks."""

    def __init__(
        self,
        cache: CandidateCache | SemanticCache,
        reranker: CandidateReranker | NliReranker | None = None,
        *,
        cosine_threshold: float = _DEFAULT_COSINE_THRESHOLD,
        tenant_cosine_thresholds: Mapping[str, float] | None = None,
        stage0_enabled: bool = False,
        nli_enabled: bool = False,
        metrics_hook: MetricHook | None = None,
    ) -> None:
        self._validate_threshold(cosine_threshold)
        tenant_thresholds = dict(tenant_cosine_thresholds or {})
        for threshold in tenant_thresholds.values():
            self._validate_threshold(threshold)
        if nli_enabled and reranker is None:
            raise ValueError("reranker is required when nli_enabled is true")
        self._cache = cache
        self._reranker = reranker
        self._cosine_threshold = cosine_threshold
        self._tenant_cosine_thresholds = tenant_thresholds
        self._stage0_enabled = stage0_enabled
        self._nli_enabled = nli_enabled
        self._metrics_hook = metrics_hook

    async def lookup(self, user_content: str, key: CacheKey) -> LookupResult:
        """Return the best candidate accepted by the enabled lookup checks."""
        retrieved = await self._cache.search(
            user_content=user_content,
            key=key,
            limit=_MAX_CANDIDATES,
        )
        candidates = sorted(
            retrieved,
            key=lambda candidate: candidate.similarity,
            reverse=True,
        )[:_MAX_CANDIDATES]
        if not candidates:
            self._emit(
                LookupMetric(
                    stage="cosine",
                    outcome="no_candidates",
                    candidate_rank=None,
                )
            )
            return LookupResult(entry=None)

        cosine_candidates: list[tuple[int, CacheEntry]] = []
        threshold = self._tenant_cosine_thresholds.get(
            key.tenant_id,
            self._cosine_threshold,
        )
        for rank, candidate in enumerate(candidates, start=1):
            passed = candidate.similarity >= threshold
            self._emit(
                LookupMetric(
                    stage="cosine",
                    outcome="accepted" if passed else "rejected",
                    candidate_rank=rank,
                    cosine_similarity=candidate.similarity,
                )
            )
            if passed:
                cosine_candidates.append((rank, candidate))

        guarded_candidates: list[tuple[int, CacheEntry]] = []
        for rank, candidate in cosine_candidates:
            if self._stage0_enabled:
                passed = stage0_guard(candidate.user_content, user_content)
                self._emit(
                    LookupMetric(
                        stage="stage0",
                        outcome="accepted" if passed else "rejected",
                        candidate_rank=rank,
                        cosine_similarity=candidate.similarity,
                    )
                )
            else:
                passed = True
            if passed:
                guarded_candidates.append((rank, candidate))
        if not guarded_candidates:
            return LookupResult(entry=None)
        if not self._nli_enabled:
            return LookupResult(entry=guarded_candidates[0][1])

        if self._reranker is None:
            raise RuntimeError("NLI is enabled without a reranker")
        candidate_ranks = {
            id(candidate): rank for rank, candidate in guarded_candidates
        }
        decisions = await self._reranker.rerank(
            user_content,
            [candidate for _, candidate in guarded_candidates],
        )
        decisions_by_rank: dict[int, NliDecision] = {}
        for decision in decisions:
            candidate_rank = candidate_ranks.get(id(decision.candidate))
            if candidate_rank is None or candidate_rank in decisions_by_rank:
                raise RuntimeError("NLI reranker returned an unexpected candidate")
            decisions_by_rank[candidate_rank] = decision
        if len(decisions_by_rank) != len(guarded_candidates):
            raise RuntimeError("NLI reranker omitted one or more candidates")

        for rank, candidate in guarded_candidates:
            decision = decisions_by_rank[rank]
            self._emit(
                LookupMetric(
                    stage="nli",
                    outcome="accepted" if decision.accepted else "rejected",
                    candidate_rank=rank,
                    cosine_similarity=candidate.similarity,
                    candidate_to_query_entailment=decision.candidate_to_query,
                    query_to_candidate_entailment=decision.query_to_candidate,
                )
            )
        for rank, _ in guarded_candidates:
            decision = decisions_by_rank[rank]
            if decision.accepted:
                return LookupResult(entry=decision.candidate)
        return LookupResult(entry=None)

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("cosine_threshold must be between -1 and 1")

    def _emit(self, metric: LookupMetric) -> None:
        if self._metrics_hook is not None:
            self._metrics_hook(metric)
