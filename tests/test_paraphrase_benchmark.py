"""Tests for PAWS and QQP dataset configuration."""

import pytest

from benchmarks.evaluate_paraphrase import dataset_spec


def test_paws_uses_labeled_test_split() -> None:
    assert dataset_spec("paws") == (
        "google-research-datasets/paws",
        "labeled_final",
        "test",
        "sentence1",
        "sentence2",
    )


def test_qqp_uses_labeled_validation_split() -> None:
    assert dataset_spec("qqp") == (
        "nyu-mll/glue",
        "qqp",
        "validation",
        "question1",
        "question2",
    )


def test_unknown_paraphrase_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="paws"):
        dataset_spec("unknown")
