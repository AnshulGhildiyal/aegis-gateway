"""Tests for train-calibrated validation-only paraphrase ablations."""

import asyncio
import csv
from pathlib import Path
from typing import cast

import pytest

from aegis.engine.embeddings import FastEmbedder
from aegis.engine.nli_reranker import OnnxNliScorer
from benchmarks.evaluate_paraphrase_ablation import (
    _calibration_sweep,
    _ensure_cosine_scores,
    _ensure_nli_scores,
    _load_score_cache,
    _load_split_rows,
    _pair_key,
    _reservoir_sample,
    _selected_validation_rows,
    _stage0_analysis,
    _validation_sweep,
    _write_csv,
)
from benchmarks.evaluation import PairExample, PairScore


def test_seeded_reservoir_sampling_is_repeatable_and_bounded() -> None:
    rows = [{"value": index} for index in range(100)]

    sample_one = _reservoir_sample(rows, 5, seed=42)
    sample_two = _reservoir_sample(rows, 5, seed=42)

    assert sample_one == sample_two
    assert len(sample_one) == 5


def test_split_loader_streams_only_explicit_train_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.evaluate_paraphrase_ablation as ablation

    calls: list[tuple[str, bool]] = []

    def fake_load_dataset(
        repository: str,
        config: str,
        *,
        split: str,
        streaming: bool,
    ) -> list[dict[str, object]]:
        del repository, config
        calls.append((split, streaming))
        return [{"row": index} for index in range(5)]

    monkeypatch.setattr(ablation, "load_dataset", fake_load_dataset)
    train_rows = asyncio.run(
        _load_split_rows("repo", "config", "train", limit=3)
    )
    validation_rows = asyncio.run(
        _load_split_rows("repo", "config", "validation", limit=2)
    )

    assert len(train_rows) == 3
    assert [row["row"] for row in validation_rows] == [0, 1]
    assert calls == [("train", True), ("validation", True)]


def test_split_loader_refuses_test_before_loading_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.evaluate_paraphrase_ablation as ablation

    def unexpected_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("dataset loader must not be called for test split")

    monkeypatch.setattr(ablation, "load_dataset", unexpected_load)
    with pytest.raises(ValueError, match="only train and validation"):
        asyncio.run(_load_split_rows("repo", "config", "test", limit=None))


def test_calibration_sweep_selects_maximum_recall_under_precision_floor() -> None:
    labels = [True, True, False, False]
    cosine_scores = [0.96, 0.96, 0.96, 0.96]
    stage0_passes = [True] * 4
    probabilities = [
        (0.99, 0.99),
        (0.90, 0.90),
        (0.92, 0.92),
        (0.10, 0.10),
    ]

    sweep, selected = _calibration_sweep(
        labels,
        cosine_scores,
        stage0_passes,
        probabilities,
    )

    assert len(sweep) == 2 * 99
    target_90 = next(
        row
        for row in selected
        if row["cosine_threshold"] == 0.95 and row["precision_target"] == 0.90
    )
    target_95 = next(
        row
        for row in selected
        if row["cosine_threshold"] == 0.95 and row["precision_target"] == 0.95
    )
    assert target_90["exists"] is True
    assert target_90["selected_nli_threshold"] == 0.99
    assert target_90["recall"] == 0.5
    assert target_95["exists"] is True
    assert target_95["selected_nli_threshold"] == 0.99


def test_calibration_reports_no_threshold_meeting_precision_target() -> None:
    labels = [True, False]
    _, selected = _calibration_sweep(
        labels,
        [0.96, 0.96],
        [True, True],
        [(0.8, 0.8), (0.9, 0.9)],
    )

    no_95 = [
        row
        for row in selected
        if row["precision_target"] == 0.95 and row["cosine_threshold"] == 0.85
    ]
    assert no_95[0]["exists"] is False
    assert no_95[0]["selected_nli_threshold"] is None


def test_validation_sweep_reports_baselines_and_nli_threshold_curve() -> None:
    rows = _validation_sweep(
        [True, False],
        [0.96, 0.96],
        [True, True],
        [(0.8, 0.8), (0.9, 0.9)],
    )

    assert len(rows) == 2 * (2 + 99)
    at_085 = [row for row in rows if row["cosine_threshold"] == 0.85]
    assert sum(row["configuration"] == "cosine_only" for row in at_085) == 1
    assert sum(row["configuration"] == "cosine_stage0" for row in at_085) == 1
    assert sum(row["configuration"] == "cosine_stage0_nli" for row in at_085) == 99
    assert next(
        row for row in at_085
        if row["configuration"] == "cosine_stage0_nli"
    )["nli_pairs_sent"] == 2


def test_selected_validation_rows_use_train_selected_threshold() -> None:
    train_selection = [
        {
            "cosine_threshold": 0.85,
            "precision_target": 0.9,
            "exists": True,
            "selected_nli_threshold": 0.75,
        }
    ]
    validation = [
        {
            "cosine_threshold": 0.85,
            "configuration": "cosine_stage0_nli",
            "nli_threshold": 0.75,
            "hits": 2,
            "true_positive": 2,
            "false_positive": 0,
            "false_negative": 1,
            "precision": 1.0,
            "recall": 2 / 3,
            "f1": 0.8,
            "nli_pairs_sent": 3,
            "nli_pairs_sent_percent": 30.0,
        }
    ]

    result = _selected_validation_rows(train_selection, validation)

    assert result[0]["selected_nli_threshold"] == 0.75
    assert result[0]["precision"] == 1.0
    assert result[0]["nli_pairs_sent"] == 3


def test_score_cache_loads_latest_record_per_pair(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"key":"abc","cosine_similarity":0.8}\n'
        '{"key":"abc","cosine_similarity":0.9,"nli_probabilities":[0.7,0.8]}\n',
        encoding="utf-8",
    )

    cache = _load_score_cache(path)

    assert cache == {
        "abc": {
            "key": "abc",
            "cosine_similarity": 0.9,
            "nli_probabilities": [0.7, 0.8],
        }
    }


def test_score_cache_reuses_cosine_and_computes_only_missing(tmp_path: Path) -> None:
    examples = [
        PairExample("cached", "pair", True),
        PairExample("new", "pair", False),
    ]
    cache: dict[str, dict[str, object]] = {
        _pair_key(examples[0], "model"): {"cosine_similarity": 0.75},
    }

    class FakeEmbedder:
        model_name = "model"
        calls = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            assert texts == ["new", "pair"]
            return [[1.0, 0.0], [1.0, 0.0]]

    embedder = FakeEmbedder()
    scores, reused, computed = asyncio.run(
        _ensure_cosine_scores(
            examples,
            cast(FastEmbedder, embedder),
            cache,
            tmp_path / "scores.jsonl",
        )
    )

    assert [score.cosine_similarity for score in scores] == [0.75, 1.0]
    assert (reused, computed, embedder.calls) == (1, 1, 1)


def test_score_cache_computes_nli_only_for_missing_pairs(tmp_path: Path) -> None:
    examples = [
        PairExample("cached", "eligible", True),
        PairExample("new", "eligible", False),
        PairExample("blocked 5 days", "blocked 15 days", True),
    ]
    scores = [PairScore(example, 0.9) for example in examples]
    cache = {
        _pair_key(examples[0], "model"): {
            "cosine_similarity": 0.9,
            "nli_probabilities": [0.8, 0.85],
        }
    }

    class FakeScorer:
        pairs: list[tuple[str, str]] = []

        async def entailment_probabilities(
            self,
            pairs: list[tuple[str, str]],
        ) -> list[float]:
            self.pairs.extend(pairs)
            return [0.6, 0.7]

    scorer = FakeScorer()
    probabilities, reused, computed = asyncio.run(
        _ensure_nli_scores(
            examples,
            scores,
            [(), (), ("number_mismatch",)],
            "model",
            cast(OnnxNliScorer, scorer),
            cache,
            tmp_path / "scores.jsonl",
        )
    )

    assert probabilities == [(0.8, 0.85), (0.6, 0.7), None]
    assert (reused, computed) == (1, 1)
    assert scorer.pairs == [("new", "eligible"), ("eligible", "new")]


def test_stage0_breakdown_and_true_paraphrase_sample_are_seeded() -> None:
    examples = [
        PairExample("Wait 5 days.", "Wait 15 days.", True),
        PairExample("Do not wait 5 days.", "Wait 15 days.", False),
        PairExample("Cancel it.", "Don't cancel it.", True),
        PairExample("No conflict.", "No conflict.", False),
    ]
    reasons = [
        ("number_mismatch",),
        ("number_mismatch", "negation_mismatch"),
        ("negation_mismatch",),
        (),
    ]
    scores = [PairScore(example, 0.9) for example in examples]

    breakdown, sample = _stage0_analysis(
        examples,
        scores,
        reasons,
        dataset_name="paws",
    )

    lookup = {
        (row["reason"], row["gold_label"]): row["rejections"] for row in breakdown
    }
    assert len(breakdown) == 4
    assert lookup[("number_mismatch", 1)] == 1
    assert lookup[("number_mismatch", 0)] == 1
    assert lookup[("negation_mismatch", 1)] == 1
    assert lookup[("negation_mismatch", 0)] == 1
    assert len(sample) == 2
    assert {row["dataset_row_index"] for row in sample} == {0, 2}


def test_csv_writer_adds_reproduction_metadata(tmp_path: Path) -> None:
    path = tmp_path / "report.csv"
    _write_csv(
        path,
        ("dataset", "split", "threshold"),
        [{"threshold": 0.85}],
        {"dataset": "paws", "split": "validation"},
    )

    with path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row == {
        "dataset": "paws",
        "split": "validation",
        "threshold": "0.85",
    }
