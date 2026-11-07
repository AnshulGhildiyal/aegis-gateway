"""Tests for validation-only PAWS and QQP ablation reporting."""

import csv
import json
from pathlib import Path

import pytest

from benchmarks.evaluate_paraphrase_ablation import (
    _ablation_rows,
    _metrics,
    _save_outputs,
    select_nli_threshold,
)
from benchmarks.evaluation import PairExample, PairScore


def test_select_nli_threshold_uses_mean_f1_and_conservative_tie_break() -> None:
    threshold, curve = select_nli_threshold(
        [True, False],
        [0.9, 0.9],
        [True, True],
        [(0.8, 0.8), (0.8, 0.8)],
        cosine_thresholds=[0.85],
        nli_thresholds=[0.7, 0.8, 0.9],
    )

    assert threshold == 0.8
    assert curve == [
        {"nli_threshold": 0.7, "mean_f1": 2 / 3},
        {"nli_threshold": 0.8, "mean_f1": 2 / 3},
        {"nli_threshold": 0.9, "mean_f1": 0.0},
    ]


def test_select_nli_threshold_rejects_mismatched_calibration_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        select_nli_threshold([True], [], [True], [None])


def test_ablation_metrics_and_nli_sent_counts_follow_stage_order() -> None:
    examples = [
        PairExample("high positive", "high paraphrase", True),
        PairExample("high negative", "high unrelated", False),
        PairExample("guard mismatch 5 days", "guard mismatch 15 days", True),
        PairExample("low positive", "low paraphrase", True),
    ]
    scores = [
        PairScore(example, similarity)
        for example, similarity in zip(
            examples,
            [0.96, 0.93, 0.91, 0.86],
            strict=True,
        )
    ]
    rows, errors = _ablation_rows(
        examples,
        scores,
        [True, True, False, True],
        [(0.9, 0.9), (0.9, 0.9), None, (0.9, 0.9)],
        nli_threshold=0.8,
        dataset_name="test",
    )

    at_090 = [row for row in rows if row["cosine_threshold"] == 0.9]
    cosine_only = next(row for row in at_090 if row["configuration"] == "cosine_only")
    stage0 = next(row for row in at_090 if row["configuration"] == "cosine_stage0")
    full = next(row for row in at_090 if row["configuration"] == "cosine_stage0_nli")
    assert _metrics([True, False], [True, False])["f1"] == 1
    assert cosine_only["hits"] == 3
    assert stage0["hits"] == 2
    assert full["hits"] == 2
    assert full["nli_pairs_sent"] == 2
    assert full["nli_pairs_sent_percent"] == 50
    assert len(errors) == 12


def test_saved_csvs_include_reproduction_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.evaluate_paraphrase_ablation as ablation

    monkeypatch.setattr(ablation, "_RESULTS_DIRECTORY", Path(tmp_path))
    files = _save_outputs(
        "paws",
        repository="google-research-datasets/paws",
        config="labeled_final",
        split="validation",
        example_count=1,
        nli_threshold=0.8,
        calibration=[{"nli_threshold": 0.8, "mean_f1": 1.0}],
        metrics_rows=[
            {
                "cosine_threshold": 0.85,
                "configuration": "cosine_only",
                "hits": 1,
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "nli_pairs_sent": None,
                "nli_pairs_sent_percent": None,
                "nli_threshold": None,
            }
        ],
        error_rows=[],
        embedding_model_source="embedding-model-repository",
        run_date="2026-11-07T00:00:00+00:00",
        git_commit="abc123",
        git_worktree_dirty=False,
        limit=None,
    )

    with Path(files["metrics_csv"]).open(encoding="utf-8", newline="") as stream:
        metric_row = next(csv.DictReader(stream))
    with Path(files["false_positives_csv"]).open(
        encoding="utf-8",
        newline="",
    ) as stream:
        error_headers = next(csv.reader(stream))
    metadata = json.loads(Path(files["json"]).read_text())

    for row in (metric_row,):
        assert row["dataset"] == "paws"
        assert row["split"] == "validation"
        assert row["run_date_utc"] == "2026-11-07T00:00:00+00:00"
        assert row["git_commit"] == "abc123"
        assert row["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert row["nli_model"] == "Xenova/nli-deberta-v3-small"
    assert {
        "dataset",
        "split",
        "run_date_utc",
        "git_commit",
        "embedding_model",
        "nli_model",
    } <= set(error_headers)
    assert metadata["test_split_used"] is False
