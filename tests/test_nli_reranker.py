"""Tests for bidirectional entailment filtering."""

import asyncio

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from aegis.engine.nli_reranker import (
    NliReranker,
    NliScorer,
    OnnxNliScorer,
    _OnnxInput,
)
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


def test_onnx_scorer_pads_variable_length_pairs_in_batch() -> None:
    class FakeInput:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeSession:
        def get_inputs(self) -> list[_OnnxInput]:
            return [
                FakeInput("input_ids"),
                FakeInput("attention_mask"),
                FakeInput("token_type_ids"),
            ]

        def run(
            self,
            output_names: list[str] | None,
            input_feed: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            del output_names
            lengths = {array.shape[1] for array in input_feed.values()}
            assert len(lengths) == 1
            return [np.asarray([[0.0, 0.0, 1.0]] * len(input_feed["input_ids"]))]

    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "[PAD]": 0,
                "[UNK]": 1,
                "short": 2,
                "premise": 3,
                "much": 4,
                "longer": 5,
                "hypothesis": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    scorer = OnnxNliScorer()
    scorer._tokenizer = tokenizer
    scorer._session = FakeSession()
    scorer._entailment_index = 2
    try:
        probabilities = scorer._score_sync(
            [
                ("short premise", "short hypothesis"),
                ("much longer premise", "longer hypothesis"),
            ]
        )
    finally:
        asyncio.run(scorer.close())

    assert probabilities == pytest.approx([0.5761168847658291, 0.5761168847658291])


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
