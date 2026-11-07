"""Tests for STS-B fine-sweep reporting helpers."""

from benchmarks.evaluation import PairExample, PairScore, threshold_sweep
from benchmarks.stsb_fine_sweep import _error_rows, _summaries


def test_summaries_choose_best_f1_and_lowest_precision_threshold() -> None:
    labels = [True, True, True, False]
    similarities = [0.98, 0.92, 0.84, 0.85]
    rows = threshold_sweep(labels, similarities, [0.80, 0.85, 0.90, 0.95])

    best_f1, precision_95, precision_90 = _summaries(rows)

    assert best_f1.threshold == 0.8
    assert best_f1.hit_count == 4
    assert best_f1.false_positive_count == 1
    assert precision_95.threshold == 0.9
    assert precision_95.precision == 1
    assert precision_95.recall == 2 / 3
    assert precision_90.threshold == 0.9


def test_summaries_report_unmet_precision_target_as_unavailable() -> None:
    rows = threshold_sweep([True, False], [0.9, 0.95], [0.8, 0.9])

    _, precision_95, precision_90 = _summaries(rows)

    assert precision_95.threshold is None
    assert precision_90.threshold is None


def test_error_rows_apply_inclusive_fp_and_fn_thresholds() -> None:
    scores = [
        PairScore(PairExample("positive candidate", "positive query", True), 0.87),
        PairScore(PairExample("missed candidate", "missed query", True), 0.50),
        PairScore(PairExample("fp at 0.95", "query one", False), 0.95),
        PairScore(PairExample("fp at 0.92", "query two", False), 0.92),
    ]

    rows = _error_rows(scores, [True, True, False, False], [4.5, 4.0, 3.9, 1.0])

    fp_092 = [row for row in rows if row["error_group"] == "false_positive_at_0.92"]
    fp_096 = [row for row in rows if row["error_group"] == "false_positive_at_0.96"]
    fn_088 = [row for row in rows if row["error_group"] == "false_negative_at_0.88"]
    assert [row["cosine_score"] for row in fp_092] == [0.95, 0.92]
    assert fp_096 == []
    assert [row["cosine_score"] for row in fn_088] == [0.87, 0.5]
    assert fp_092[0]["gold_score"] == 3.9
