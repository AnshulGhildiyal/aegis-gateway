"""Tests for dataset evaluation utility functions."""

from collections.abc import Sequence

import pytest

from aegis.engine.nli_reranker import NliDecision
from aegis.storage.cache import CacheEntry
from benchmarks.evaluation import (
    PairExample,
    binary_metrics,
    evaluate_pair_examples,
    intent_hit_metrics,
    rows_to_intent_examples,
    rows_to_pair_examples,
    score_pair_cosines,
    stsb_binary_labels,
    threshold_sweep,
)


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "different" not in text else [0.0, 1.0] for text in texts]


class FakeReranker:
    async def rerank(
        self,
        query: str,
        candidates: Sequence[CacheEntry],
    ) -> list[NliDecision]:
        return [
            NliDecision(
                candidate=candidate,
                candidate_to_query=0.95,
                query_to_candidate=0.95,
                accepted=True,
            )
            for candidate in candidates
        ]


def test_stsb_scores_use_score_greater_than_or_equal_to_four() -> None:
    assert stsb_binary_labels([0.0, 3.99, 4.0, 5.0]) == [
        False,
        False,
        True,
        True,
    ]


def test_stsb_labels_reject_values_outside_the_dataset_scale() -> None:
    with pytest.raises(ValueError, match="0 to 5"):
        stsb_binary_labels([5.1])


def test_threshold_sweep_reports_confusion_and_rates() -> None:
    results = threshold_sweep(
        [True, True, False, False],
        [0.9, 0.4, 0.8, 0.2],
        [0.5],
    )

    assert results == [
        {
            "threshold": 0.5,
            "true_positive": 1,
            "true_negative": 1,
            "false_positive": 1,
            "false_negative": 1,
            "precision": 0.5,
            "recall": 0.5,
            "false_positive_rate": 0.5,
            "accuracy": 0.5,
        }
    ]


def test_empty_metrics_have_none_rates_instead_of_fake_values() -> None:
    metrics = binary_metrics([], [])

    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["false_positive_rate"] is None
    assert metrics["accuracy"] is None


def test_threshold_sweep_rejects_mismatched_data() -> None:
    with pytest.raises(ValueError, match="equal length"):
        threshold_sweep([True], [], [0.8])


def test_threshold_sweep_rejects_invalid_cosine_similarity() -> None:
    with pytest.raises(ValueError, match="similarities"):
        threshold_sweep([True], [float("nan")], [0.8])


def test_pair_dataset_rows_map_binary_labels() -> None:
    examples = rows_to_pair_examples(
        [
            {"a": "original", "b": "paraphrase", "label": 1},
            {"a": "one", "b": "other", "label": 0},
        ],
        text_a_field="a",
        text_b_field="b",
        label_field="label",
    )

    assert examples == [
        PairExample("original", "paraphrase", True),
        PairExample("one", "other", False),
    ]


def test_pair_dataset_limit_and_invalid_label() -> None:
    rows = [
        {"a": "first", "b": "first query", "label": 1},
        {"a": "second", "b": "second query", "label": 0},
    ]
    assert (
        len(
            rows_to_pair_examples(
                rows,
                text_a_field="a",
                text_b_field="b",
                label_field="label",
                limit=1,
            )
        )
        == 1
    )
    with pytest.raises(ValueError, match="binary"):
        rows_to_pair_examples(
            [{"a": "a", "b": "b", "label": -1}],
            text_a_field="a",
            text_b_field="b",
            label_field="label",
        )


def test_banking_rows_resolve_class_label_indices() -> None:
    examples = rows_to_intent_examples(
        [
            {"text": "Where is my card?", "label": 0},
            {"text": "Cancel my card", "label": "cash_withdrawal"},
        ],
        label_names=["cash_withdrawal"],
    )

    assert [example.intent for example in examples] == [
        "cash_withdrawal",
        "cash_withdrawal",
    ]


def test_banking_rows_reject_invalid_class_label_index() -> None:
    with pytest.raises(ValueError, match="outside"):
        rows_to_intent_examples(
            [{"text": "Question", "label": 2}],
            label_names=["one"],
        )


def test_banking_false_positive_is_a_hit_with_a_different_intent() -> None:
    metrics = intent_hit_metrics(
        ["same", "other", "same", "other"],
        ["same", "same", None, "other"],
    )

    assert metrics == {
        "hits": 3,
        "same_intent_hits": 2,
        "false_positive_hits_different_intent": 1,
        "false_positive_rate_of_hits": pytest.approx(1 / 3),
        "false_positive_rate_of_test_examples": 0.25,
    }


def test_banking_false_positive_rates_are_none_without_hits() -> None:
    metrics = intent_hit_metrics(["one"], [None])

    assert metrics["false_positive_rate_of_hits"] is None
    assert metrics["false_positive_rate_of_test_examples"] == 0


async def test_pair_cosine_scoring_reports_one_score_per_example() -> None:
    examples = [
        PairExample("same", "same query", True),
        PairExample("different candidate", "unrelated query", False),
    ]

    scores, embedding_seconds, cosine_seconds = await score_pair_cosines(
        examples,
        FakeEmbedder(),
    )

    assert [score.cosine_similarity for score in scores] == pytest.approx([1.0, 0.0])
    assert embedding_seconds >= 0
    assert len(cosine_seconds) == 2
    assert all(value >= 0 for value in cosine_seconds)


async def test_pair_eval_reports_cosine_and_full_pipeline_latency_separately() -> None:
    report = await evaluate_pair_examples(
        [PairExample("same candidate", "same query", True)],
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        cosine_threshold=0.8,
    )

    assert report.example_count == 1
    assert report.cosine_only["true_positive"] == 1
    assert report.cosine_plus_stage0_nli["true_positive"] == 1
    assert report.latency_ms["cosine_only_per_pair"]["mean"] is not None
    assert report.latency_ms["cosine_plus_stage0_nli_per_pair"]["mean"] is not None
    assert report.as_dict()["latency_ms"] == report.latency_ms


async def test_pair_eval_rejects_embedder_count_mismatch() -> None:
    class BrokenEmbedder:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return []

    with pytest.raises(RuntimeError, match="expected 2"):
        await score_pair_cosines(
            [PairExample("candidate", "query", True)],
            BrokenEmbedder(),
        )
