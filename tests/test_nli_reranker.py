"""Tests for bidirectional entailment filtering."""

import pytest

from aegis.engine.nli_reranker import NliReranker, NliScorer, OnnxNliScorer
from aegis.storage.cache import CacheEntry


class FakeNliScorer:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.pairs: list[tuple[str, str]] = []

    async def entailment_probabilities(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        self.pairs = pairs
        return self.probabilities

    async def close(self) -> None:
        return None


def candidate(content: str, similarity: float) -> CacheEntry:
    return CacheEntry(
        user_content=content,
        response={"answer": content},
        created_at=1.0,
        expires_at=2.0,
        last_accessed_at=1.0,
        similarity=similarity,
    )


async def test_candidate_is_accepted_only_when_both_directions_pass() -> None:
    scorer = FakeNliScorer([0.95, 0.91, 0.99, 0.79])
    reranker = NliReranker(scorer, entailment_threshold=0.8)

    decisions = await reranker.rerank(
        "query text",
        [candidate("accepted text", 0.9), candidate("one-way text", 0.8)],
    )

    assert len(decisions) == 2
    assert decisions[0].accepted
    assert not decisions[1].accepted
    assert decisions[1].candidate_to_query == 0.99
    assert decisions[1].query_to_candidate == 0.79


async def test_only_top_two_cosine_candidates_are_scored() -> None:
    scorer = FakeNliScorer([0.9] * 4)
    reranker = NliReranker(scorer, entailment_threshold=0.8)
    candidates = [
        candidate("third", 0.7),
        candidate("first", 0.9),
        candidate("fourth", 0.6),
        candidate("second", 0.8),
    ]

    decisions = await reranker.rerank("query", candidates)

    assert [decision.candidate.user_content for decision in decisions] == [
        "first",
        "second",
    ]
    assert scorer.pairs == [
        ("first", "query"),
        ("query", "first"),
        ("second", "query"),
        ("query", "second"),
    ]


async def test_empty_candidates_do_not_call_scorer() -> None:
    scorer = FakeNliScorer([])
    reranker = NliReranker(scorer, entailment_threshold=0.8)

    assert await reranker.rerank("query", []) == []
    assert scorer.pairs == []


async def test_scorer_must_return_two_scores_per_candidate() -> None:
    reranker = NliReranker(FakeNliScorer([0.9]), entailment_threshold=0.8)

    with pytest.raises(RuntimeError, match="expected 2"):
        await reranker.rerank("query", [candidate("candidate", 0.9)])


async def test_invalid_scorer_probability_is_rejected() -> None:
    reranker = NliReranker(FakeNliScorer([1.1, 0.9]), entailment_threshold=0.8)

    with pytest.raises(RuntimeError, match="invalid probability"):
        await reranker.rerank("query", [candidate("candidate", 0.9)])


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="entailment_threshold"):
        NliReranker(entailment_threshold=threshold)


async def test_threshold_is_inclusive() -> None:
    scorer: NliScorer = FakeNliScorer([0.8, 0.8])
    reranker = NliReranker(scorer, entailment_threshold=0.8)

    decisions = await reranker.rerank("query", [candidate("candidate", 0.9)])

    assert decisions[0].accepted


async def test_onnx_scorer_empty_pairs_do_not_download_model() -> None:
    scorer = OnnxNliScorer()
    try:
        assert await scorer.entailment_probabilities([]) == []
    finally:
        await scorer.close()


@pytest.mark.parametrize(
    ("config", "expected_index"),
    [
        ({"label2id": {"contradiction": 0, "entailment": 1}}, 1),
        ({"id2label": {"0": "contradiction", "2": "entailment"}}, 2),
    ],
)
def test_onnx_scorer_reads_entailment_label_from_model_config(
    config: object,
    expected_index: int,
) -> None:
    assert OnnxNliScorer._read_entailment_index(config) == expected_index


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"label2id": {"neutral": 0}},
        {"id2label": {"not-an-index": "entailment"}},
    ],
)
def test_onnx_scorer_rejects_config_without_entailment_index(
    config: object,
) -> None:
    with pytest.raises(RuntimeError):
        OnnxNliScorer._read_entailment_index(config)


@pytest.mark.parametrize("max_length", [2, 513])
def test_onnx_scorer_rejects_invalid_max_length(max_length: int) -> None:
    with pytest.raises(ValueError, match="max_length"):
        OnnxNliScorer(max_length=max_length)
