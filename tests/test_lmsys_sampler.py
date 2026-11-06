"""Tests for local LMSYS manual-label sampling utilities."""

import csv
import json
from pathlib import Path

import pytest

from benchmarks.sample_lmsys_for_labeling import (
    PromptPair,
    first_user_prompt,
    make_prompt_pairs,
    reservoir_sample_prompts,
    write_labeling_csv,
)


def test_extract_first_user_prompt_from_message_list() -> None:
    assert (
        first_user_prompt(
            [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "  A question  "},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Second question"},
            ]
        )
        == "A question"
    )


def test_extract_first_user_prompt_from_json_text() -> None:
    conversation = json.dumps([{"role": "user", "content": "Question"}])

    assert first_user_prompt(conversation) == "Question"


def test_conversation_without_user_turn_returns_none() -> None:
    assert first_user_prompt([{"role": "assistant", "content": "Answer"}]) is None


def test_malformed_conversation_shape_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="list"):
        first_user_prompt({"role": "user"})
    with pytest.raises(ValueError, match="messages"):
        first_user_prompt([None])


def test_reservoir_sample_is_seeded_and_respects_scan_limit() -> None:
    rows = [
        {"conversation": [{"role": "user", "content": f"Prompt {index}"}]}
        for index in range(10)
    ]

    first, scanned = reservoir_sample_prompts(
        rows,
        prompt_count=3,
        scan_limit=8,
        seed=7,
    )
    second, second_scanned = reservoir_sample_prompts(
        rows,
        prompt_count=3,
        scan_limit=8,
        seed=7,
    )

    assert first == second
    assert len(first) == 3
    assert scanned == second_scanned == 8


def test_reservoir_sample_skips_duplicate_prompts() -> None:
    rows = [
        {"conversation": [{"role": "user", "content": "Same"}]},
        {"conversation": [{"role": "user", "content": "Same"}]},
        {"conversation": [{"role": "user", "content": "Different"}]},
    ]

    prompts, _ = reservoir_sample_prompts(
        rows,
        prompt_count=10,
        scan_limit=10,
        seed=1,
    )

    assert sorted(prompts) == ["Different", "Same"]


def test_prompt_pairs_are_distinct_and_seeded() -> None:
    prompts = ["one", "two", "three", "four", "five"]

    first = make_prompt_pairs(prompts, seed=3)
    second = make_prompt_pairs(prompts, seed=3)

    assert first == second
    assert len(first) == 2
    assert all(pair.candidate_prompt != pair.query_prompt for pair in first)


def test_write_labeling_csv_leaves_manual_fields_blank(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "manual.csv"

    count = write_labeling_csv(
        [PromptPair("candidate", "query")],
        output,
    )

    with output.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert count == 1
    assert rows[0]["candidate_prompt"] == "candidate"
    assert rows[0]["query_prompt"] == "query"
    assert rows[0]["manual_equivalence_label"] == ""
    assert rows[0]["notes"] == ""
