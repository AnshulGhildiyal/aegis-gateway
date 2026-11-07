"""Train-calibrated, validation-reported PAWS and QQP ablation."""

import argparse
import asyncio
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from datasets import load_dataset  # type: ignore[import-untyped]
from fastembed import TextEmbedding
from matplotlib.lines import Line2D

from aegis.engine.embeddings import FastEmbedder
from aegis.engine.nli_reranker import OnnxNliScorer
from aegis.engine.regex_guard import stage0_rejection_reasons
from benchmarks.evaluation import PairExample, PairScore, binary_metrics

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIRECTORY = _REPOSITORY_ROOT / "benchmarks" / "results"
_COSINE_THRESHOLDS = (0.85, 0.95)
_COSINE_SWEEP_THRESHOLDS = tuple(value / 100 for value in range(85, 100))
_NLI_THRESHOLDS = tuple(value / 100 for value in range(1, 100))
_PRECISION_TARGETS = (0.90, 0.95)
_MATCHED_PRECISION_TARGETS = (0.80, 0.90, 0.95)
_TRAIN_SAMPLE_SIZE = 5_000
_SAMPLE_SEED = 42
_EXAMPLES_PER_BATCH = 256
_NLI_PAIRS_PER_BATCH = 128
_STAGE0_SAMPLE_SIZE = 20
_FALSE_POSITIVE_SAMPLE_SIZE = 15
_NLI_MODEL_NAME = "Xenova/nli-deberta-v3-small"
_NLI_MODEL_REVISION = "6bc2a55c7c0f7e2bc68de60bb248e523e2612abb"
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_DATASETS: dict[str, tuple[str, str, str, str]] = {
    "paws": (
        "google-research-datasets/paws",
        "labeled_final",
        "sentence1",
        "sentence2",
    ),
    "qqp": ("nyu-mll/glue", "qqp", "question1", "question2"),
}


def _pair_key(example: PairExample, embedding_model: str) -> str:
    payload = json.dumps(
        [
            embedding_model,
            _NLI_MODEL_NAME,
            _NLI_MODEL_REVISION,
            version("fastembed"),
            example.candidate,
            example.query,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_score_cache(path: Path) -> dict[str, dict[str, object]]:
    scores: dict[str, dict[str, object]] = {}
    if not path.exists():
        return scores
    with path.open(encoding="utf-8") as cache_file:
        for line_number, line in enumerate(cache_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("key"), str):
                raise ValueError(f"invalid score-cache record at {path}:{line_number}")
            scores[record["key"]] = record
    return scores


def _append_score_cache(
    path: Path,
    updates: Mapping[str, dict[str, object]],
) -> None:
    if not updates:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as cache_file:
        for key, record in updates.items():
            cache_file.write(
                json.dumps({"key": key, **record}, sort_keys=True) + "\n"
            )


def _seed_cache_from_previous_samples(
    dataset_name: str,
    score_cache: dict[str, dict[str, object]],
) -> int:
    """Reuse pair-level scores preserved in the prior ablation's error samples."""
    path = (
        _RESULTS_DIRECTORY
        / f"paraphrase_ablation_{dataset_name}_validation_false_positives.csv"
    )
    if not path.exists():
        return 0
    seeded_keys: set[str] = set()
    with path.open(encoding="utf-8", newline="") as sample_file:
        for row in csv.DictReader(sample_file):
            candidate = row.get("candidate")
            query = row.get("query")
            if not candidate or not query:
                continue
            example = PairExample(candidate, query, row.get("gold_label") == "1")
            key = _pair_key(example, _EMBEDDING_MODEL_NAME)
            record = score_cache.setdefault(key, {"cosine_similarity": None})
            cosine = row.get("cosine_similarity")
            if cosine:
                record["cosine_similarity"] = float(cosine)
            forward = row.get("candidate_to_query_entailment")
            reverse = row.get("query_to_candidate_entailment")
            if (
                forward
                and reverse
                and record.get("nli_probabilities") is None
            ):
                record["nli_probabilities"] = [float(forward), float(reverse)]
                seeded_keys.add(key)
    if seeded_keys:
        _append_score_cache(
            _RESULTS_DIRECTORY
            / f"paraphrase_ablation_{dataset_name}_score_cache.jsonl",
            {key: score_cache[key] for key in seeded_keys},
        )
    return len(seeded_keys)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _metrics(
    labels: Sequence[bool],
    predictions: Sequence[bool],
) -> dict[str, int | float | None]:
    result = binary_metrics(labels, predictions)
    return {
        "hits": cast(int, result["true_positive"])
        + cast(int, result["false_positive"]),
        "true_positive": result["true_positive"],
        "false_positive": result["false_positive"],
        "false_negative": result["false_negative"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": _f1(
            cast(float | None, result["precision"]),
            cast(float | None, result["recall"]),
        ),
    }


def _predictions(
    cosine_scores: Sequence[float],
    stage0_passes: Sequence[bool],
    bidirectional_probabilities: Sequence[tuple[float, float] | None],
    cosine_threshold: float,
    nli_threshold: float,
) -> list[bool]:
    return [
        cosine >= cosine_threshold
        and stage0_pass
        and probabilities is not None
        and probabilities[0] >= nli_threshold
        and probabilities[1] >= nli_threshold
        for cosine, stage0_pass, probabilities in zip(
            cosine_scores,
            stage0_passes,
            bidirectional_probabilities,
            strict=True,
        )
    ]


def _calibration_sweep(
    labels: Sequence[bool],
    cosine_scores: Sequence[float],
    stage0_passes: Sequence[bool],
    bidirectional_probabilities: Sequence[tuple[float, float] | None],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sweep_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for cosine_threshold in _COSINE_THRESHOLDS:
        threshold_rows: list[dict[str, object]] = []
        for nli_threshold in _NLI_THRESHOLDS:
            metrics = _metrics(
                labels,
                _predictions(
                    cosine_scores,
                    stage0_passes,
                    bidirectional_probabilities,
                    cosine_threshold,
                    nli_threshold,
                ),
            )
            row = {
                "cosine_threshold": cosine_threshold,
                "nli_threshold": nli_threshold,
                **metrics,
            }
            threshold_rows.append(cast(dict[str, object], row))
            sweep_rows.append(cast(dict[str, object], row))

        for precision_target in _PRECISION_TARGETS:
            eligible = [
                row
                for row in threshold_rows
                if row["precision"] is not None
                and cast(float, row["precision"]) >= precision_target
            ]
            selected = (
                max(
                    eligible,
                    key=lambda row: (
                        cast(float | None, row["recall"]) or 0.0,
                        cast(float, row["nli_threshold"]),
                    ),
                )
                if eligible
                else None
            )
            selected_rows.append(
                {
                    "cosine_threshold": cosine_threshold,
                    "precision_target": precision_target,
                    "exists": selected is not None,
                    "selected_nli_threshold": (
                        selected["nli_threshold"] if selected is not None else None
                    ),
                    "tie_break": "higher NLI threshold",
                    **(
                        {
                            key: selected[key]
                            for key in (
                                "hits",
                                "true_positive",
                                "false_positive",
                                "false_negative",
                                "precision",
                                "recall",
                                "f1",
                            )
                        }
                        if selected is not None
                        else {}
                    ),
                }
            )
    return sweep_rows, selected_rows


def _validation_sweep(
    labels: Sequence[bool],
    cosine_scores: Sequence[float],
    stage0_passes: Sequence[bool],
    bidirectional_probabilities: Sequence[tuple[float, float] | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cosine_threshold in _COSINE_THRESHOLDS:
        candidate_indices = [
            cosine >= cosine_threshold and passed
            for cosine, passed in zip(cosine_scores, stage0_passes, strict=True)
        ]
        cosine_predictions = [
            cosine >= cosine_threshold for cosine in cosine_scores
        ]
        stage0_predictions = [
            cosine >= cosine_threshold and passed
            for cosine, passed in zip(cosine_scores, stage0_passes, strict=True)
        ]
        for configuration, metrics in (
            ("cosine_only", _metrics(labels, cosine_predictions)),
            ("cosine_stage0", _metrics(labels, stage0_predictions)),
        ):
            rows.append(
                {
                    "cosine_threshold": cosine_threshold,
                    "configuration": configuration,
                    "nli_threshold": None,
                    **metrics,
                    "nli_pairs_sent": None,
                    "nli_pairs_sent_percent": None,
                }
            )
        for nli_threshold in _NLI_THRESHOLDS:
            rows.append(
                {
                    "cosine_threshold": cosine_threshold,
                    "configuration": "cosine_stage0_nli",
                    "nli_threshold": nli_threshold,
                    **_metrics(
                        labels,
                        _predictions(
                            cosine_scores,
                            stage0_passes,
                            bidirectional_probabilities,
                            cosine_threshold,
                            nli_threshold,
                        ),
                    ),
                    "nli_pairs_sent": sum(candidate_indices),
                    "nli_pairs_sent_percent": (
                        sum(candidate_indices) / len(labels) * 100 if labels else 0.0
                    ),
                }
            )
    return rows


def _cosine_validation_sweep(
    labels: Sequence[bool],
    cosine_scores: Sequence[float],
    stage0_passes: Sequence[bool],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cosine_threshold in _COSINE_SWEEP_THRESHOLDS:
        cosine_only = [
            score >= cosine_threshold for score in cosine_scores
        ]
        cosine_stage0 = [
            score >= cosine_threshold and passed
            for score, passed in zip(cosine_scores, stage0_passes, strict=True)
        ]
        for configuration, predictions in (
            ("cosine_only", cosine_only),
            ("cosine_stage0", cosine_stage0),
        ):
            rows.append(
                {
                    "cosine_threshold": cosine_threshold,
                    "configuration": configuration,
                    **_metrics(labels, predictions),
                }
            )
    return rows


def _matched_precision_rows(
    train_labels: Sequence[bool],
    train_cosine_scores: Sequence[float],
    train_stage0_passes: Sequence[bool],
    train_probabilities: Sequence[tuple[float, float] | None],
    validation_labels: Sequence[bool],
    validation_cosine_scores: Sequence[float],
    validation_stage0_passes: Sequence[bool],
    validation_probabilities: Sequence[tuple[float, float] | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_configs = ("cosine_only", "cosine_stage0")
    for precision_target in _MATCHED_PRECISION_TARGETS:
        for configuration in baseline_configs:
            candidates: list[dict[str, object]] = []
            for cosine_threshold in _COSINE_SWEEP_THRESHOLDS:
                predictions = [
                    cosine >= cosine_threshold
                    and (
                        configuration == "cosine_only"
                        or stage0_pass
                    )
                    for cosine, stage0_pass in zip(
                        train_cosine_scores,
                        train_stage0_passes,
                        strict=True,
                    )
                ]
                candidates.append(
                    {
                        "cosine_threshold": cosine_threshold,
                        "nli_threshold": None,
                        **_metrics(train_labels, predictions),
                    }
                )
            selected = _best_precision_candidate(candidates, precision_target)
            rows.append(
                _matched_precision_result(
                    precision_target,
                    configuration,
                    selected,
                    validation_labels,
                    validation_cosine_scores,
                    validation_stage0_passes,
                    validation_probabilities,
                )
            )

        for cosine_threshold in _COSINE_THRESHOLDS:
            candidates = []
            for nli_threshold in _NLI_THRESHOLDS:
                candidates.append(
                    {
                        "cosine_threshold": cosine_threshold,
                        "nli_threshold": nli_threshold,
                        **_metrics(
                            train_labels,
                            _predictions(
                                train_cosine_scores,
                                train_stage0_passes,
                                train_probabilities,
                                cosine_threshold,
                                nli_threshold,
                            ),
                        ),
                    }
                )
            selected = _best_precision_candidate(candidates, precision_target)
            rows.append(
                _matched_precision_result(
                    precision_target,
                    "cosine_stage0_nli",
                    selected,
                    validation_labels,
                    validation_cosine_scores,
                    validation_stage0_passes,
                    validation_probabilities,
                    calibration_cosine_threshold=cosine_threshold,
                )
            )
    return rows


def _best_precision_candidate(
    candidates: Sequence[dict[str, object]],
    precision_target: float,
) -> dict[str, object] | None:
    eligible = [
        row
        for row in candidates
        if row["precision"] is not None
        and cast(float, row["precision"]) >= precision_target
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            cast(float | None, row["recall"]) or 0.0,
            cast(float, row["cosine_threshold"]),
            cast(float, row["nli_threshold"])
            if row["nli_threshold"] is not None
            else 0.0,
        ),
    )


def _matched_precision_result(
    precision_target: float,
    configuration: str,
    selected: dict[str, object] | None,
    validation_labels: Sequence[bool],
    validation_cosine_scores: Sequence[float],
    validation_stage0_passes: Sequence[bool],
    validation_probabilities: Sequence[tuple[float, float] | None],
    *,
    calibration_cosine_threshold: float | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "precision_target": precision_target,
        "configuration": configuration,
        "calibration_status": "selected" if selected is not None else "unreachable",
        "cosine_threshold": calibration_cosine_threshold,
    }
    if selected is None:
        return result

    cosine_threshold = cast(float, selected["cosine_threshold"])
    nli_threshold = cast(float | None, selected["nli_threshold"])
    if configuration == "cosine_only":
        validation_predictions = [
            cosine >= cosine_threshold for cosine in validation_cosine_scores
        ]
    elif configuration == "cosine_stage0":
        validation_predictions = [
            cosine >= cosine_threshold and stage0_pass
            for cosine, stage0_pass in zip(
                validation_cosine_scores,
                validation_stage0_passes,
                strict=True,
            )
        ]
    else:
        if nli_threshold is None:
            raise RuntimeError("NLI calibration selection is missing its threshold")
        validation_predictions = _predictions(
            validation_cosine_scores,
            validation_stage0_passes,
            validation_probabilities,
            cosine_threshold,
            nli_threshold,
        )
    validation_metrics = _metrics(validation_labels, validation_predictions)
    return {
        **result,
        "cosine_threshold": cosine_threshold,
        "nli_threshold": nli_threshold,
        "train_hits": selected["hits"],
        "train_precision": selected["precision"],
        "train_recall": selected["recall"],
        "validation_hits": validation_metrics["hits"],
        "validation_true_positive": validation_metrics["true_positive"],
        "validation_false_positive": validation_metrics["false_positive"],
        "validation_false_negative": validation_metrics["false_negative"],
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1": validation_metrics["f1"],
    }


def _stage0_candidate_set_analysis(
    examples: Sequence[PairExample],
    pair_scores: Sequence[PairScore],
    rejection_reasons: Sequence[tuple[str, ...]],
) -> list[dict[str, object]]:
    candidate_sets: list[tuple[str, float | None]] = [
        ("all_validation", None),
        *[
            (f"cosine_gte_{threshold:.2f}", threshold)
            for threshold in _COSINE_SWEEP_THRESHOLDS
        ],
    ]
    rows: list[dict[str, object]] = []
    for candidate_set, cosine_threshold in candidate_sets:
        indices = [
            index
            for index, score in enumerate(pair_scores)
            if cosine_threshold is None
            or score.cosine_similarity >= cosine_threshold
        ]
        counts: Counter[tuple[str, int]] = Counter()
        for index in indices:
            for reason in rejection_reasons[index]:
                counts[(reason, int(examples[index].label))] += 1
        for reason in ("number_mismatch", "negation_mismatch"):
            for gold_label in (0, 1):
                rows.append(
                    {
                        "candidate_set": candidate_set,
                        "cosine_threshold": cosine_threshold,
                        "candidate_count": len(indices),
                        "reason": reason,
                        "gold_label": gold_label,
                        "gold_class": (
                            "paraphrase" if gold_label else "non_paraphrase"
                        ),
                        "rejections": counts[(reason, gold_label)],
                        "counts_can_overlap": True,
                    }
                )
    return rows


def _selected_validation_rows(
    selected_thresholds: Sequence[Mapping[str, object]],
    validation_sweep: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    selected_rows: list[dict[str, object]] = []
    for selected in selected_thresholds:
        if not selected["exists"]:
            selected_rows.append(
                {
                    "cosine_threshold": selected["cosine_threshold"],
                    "precision_target": selected["precision_target"],
                    "exists": False,
                    "selected_nli_threshold": None,
                }
            )
            continue
        matching = next(
            row
            for row in validation_sweep
            if row["configuration"] == "cosine_stage0_nli"
            and row["cosine_threshold"] == selected["cosine_threshold"]
            and row["nli_threshold"] == selected["selected_nli_threshold"]
        )
        selected_rows.append(
            {
                "cosine_threshold": selected["cosine_threshold"],
                "precision_target": selected["precision_target"],
                "exists": True,
                "selected_nli_threshold": selected["selected_nli_threshold"],
                **{
                    key: matching[key]
                    for key in (
                        "hits",
                        "true_positive",
                        "false_positive",
                        "false_negative",
                        "precision",
                        "recall",
                        "f1",
                    )
                },
                "nli_pairs_sent": matching["nli_pairs_sent"],
                "nli_pairs_sent_percent": matching["nli_pairs_sent_percent"],
            }
        )
    return selected_rows


def _reservoir_sample(
    rows: Iterable[Mapping[str, object]],
    sample_size: int,
    *,
    seed: int,
) -> list[Mapping[str, object]]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    rng = random.Random(seed)
    reservoir: list[Mapping[str, object]] = []
    seen = 0
    for row in rows:
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(row)
        else:
            index = rng.randrange(seen)
            if index < sample_size:
                reservoir[index] = row
    return reservoir


def _pair_examples(
    rows: Iterable[Mapping[str, object]],
    *,
    candidate_field: str,
    query_field: str,
) -> list[PairExample]:
    examples: list[PairExample] = []
    for row in rows:
        candidate = row.get(candidate_field)
        query = row.get(query_field)
        label = row.get("label")
        if not isinstance(candidate, str) or not isinstance(query, str):
            raise ValueError("dataset pair fields must be strings")
        if isinstance(label, bool):
            positive = label
        elif isinstance(label, int) and label in (0, 1):
            positive = label == 1
        else:
            raise ValueError("dataset label must be binary 0/1")
        examples.append(PairExample(candidate, query, positive))
    return examples


async def _ensure_cosine_scores(
    examples: Sequence[PairExample],
    embedder: FastEmbedder,
    score_cache: dict[str, dict[str, object]],
    cache_path: Path,
) -> tuple[list[PairScore], int, int]:
    keys = [_pair_key(example, embedder.model_name) for example in examples]
    missing_indices = [
        index
        for index, key in enumerate(keys)
        if score_cache.get(key, {}).get("cosine_similarity") is None
    ]
    reused_count = len(examples) - len(missing_indices)
    for offset in range(0, len(missing_indices), _EXAMPLES_PER_BATCH):
        indices = missing_indices[offset : offset + _EXAMPLES_PER_BATCH]
        batch_examples = [examples[index] for index in indices]
        texts = [
            text
            for example in batch_examples
            for text in (example.candidate, example.query)
        ]
        vectors = await embedder.embed(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors; expected {len(texts)}"
            )
        batch_updates: dict[str, dict[str, object]] = {}
        for local_index, dataset_index in enumerate(indices):
            candidate_vector = vectors[local_index * 2]
            query_vector = vectors[local_index * 2 + 1]
            if not all(
                math.isfinite(value) for value in (*candidate_vector, *query_vector)
            ):
                raise RuntimeError("embedder returned a non-finite vector")
            candidate_norm = sum(value * value for value in candidate_vector) ** 0.5
            query_norm = sum(value * value for value in query_vector) ** 0.5
            if candidate_norm == 0 or query_norm == 0:
                similarity = 0.0
            else:
                similarity = sum(
                    first * second
                    for first, second in zip(
                        candidate_vector,
                        query_vector,
                        strict=True,
                    )
                ) / (candidate_norm * query_norm)
            similarity = min(1.0, max(-1.0, similarity))
            key = keys[dataset_index]
            previous = score_cache.get(key, {})
            record = {
                **previous,
                "cosine_similarity": similarity,
            }
            score_cache[key] = record
            batch_updates[key] = record
        _append_score_cache(cache_path, batch_updates)

    return (
        [
            PairScore(
                example,
                cast(float, score_cache[key]["cosine_similarity"]),
            )
            for example, key in zip(examples, keys, strict=True)
        ],
        reused_count,
        len(missing_indices),
    )


async def _ensure_nli_scores(
    examples: Sequence[PairExample],
    pair_scores: Sequence[PairScore],
    stage0_reasons: Sequence[tuple[str, ...]],
    embedder_model: str,
    scorer: OnnxNliScorer,
    score_cache: dict[str, dict[str, object]],
    cache_path: Path,
) -> tuple[list[tuple[float, float] | None], int, int]:
    keys = [_pair_key(example, embedder_model) for example in examples]
    eligible_indices = [
        index
        for index, (score, reasons) in enumerate(
            zip(pair_scores, stage0_reasons, strict=True)
        )
        if score.cosine_similarity >= min(_COSINE_THRESHOLDS) and not reasons
    ]
    missing_indices = [
        index
        for index in eligible_indices
        if score_cache.get(keys[index], {}).get("nli_probabilities") is None
    ]
    reused_count = len(eligible_indices) - len(missing_indices)
    for offset in range(0, len(missing_indices), _NLI_PAIRS_PER_BATCH):
        indices = missing_indices[offset : offset + _NLI_PAIRS_PER_BATCH]
        pairs = [
            pair
            for index in indices
            for pair in (
                (examples[index].candidate, examples[index].query),
                (examples[index].query, examples[index].candidate),
            )
        ]
        values = await scorer.entailment_probabilities(pairs)
        if len(values) != len(pairs):
            raise RuntimeError(
                f"NLI scorer returned {len(values)} scores; expected {len(pairs)}"
            )
        if any(not 0 <= value <= 1 for value in values):
            raise RuntimeError("NLI scorer returned an invalid probability")
        updates: dict[str, dict[str, object]] = {}
        for local_index, dataset_index in enumerate(indices):
            key = keys[dataset_index]
            record = {
                **score_cache.get(key, {}),
                "nli_probabilities": [
                    values[local_index * 2],
                    values[local_index * 2 + 1],
                ],
            }
            score_cache[key] = record
            updates[key] = record
        _append_score_cache(cache_path, updates)

    eligible_set = set(eligible_indices)
    probabilities: list[tuple[float, float] | None] = []
    for index, key in enumerate(keys):
        raw = score_cache.get(key, {}).get("nli_probabilities")
        if index not in eligible_set or not isinstance(raw, list):
            probabilities.append(None)
        elif len(raw) != 2:
            raise ValueError(f"invalid cached NLI probability pair for key {key}")
        else:
            pair = (float(raw[0]), float(raw[1]))
            if any(not 0 <= value <= 1 or not math.isfinite(value) for value in pair):
                raise ValueError(f"invalid cached NLI probabilities for key {key}")
            probabilities.append(pair)
    return probabilities, reused_count, len(missing_indices)


def _stage0_analysis(
    examples: Sequence[PairExample],
    pair_scores: Sequence[PairScore],
    rejection_reasons: Sequence[tuple[str, ...]],
    *,
    dataset_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    counts: Counter[tuple[str, int]] = Counter()
    rejected_true: list[int] = []
    for index, (example, reasons) in enumerate(
        zip(examples, rejection_reasons, strict=True)
    ):
        for reason in reasons:
            counts[(reason, int(example.label))] += 1
        if reasons and example.label:
            rejected_true.append(index)

    breakdown: list[dict[str, object]] = [
        {
            "reason": reason,
            "gold_label": gold_label,
            "gold_class": "paraphrase" if gold_label else "non_paraphrase",
            "rejections": counts[(reason, gold_label)],
            "counts_can_overlap": True,
        }
        for reason in ("number_mismatch", "negation_mismatch")
        for gold_label in (0, 1)
    ]
    sampled = random.Random(f"{_SAMPLE_SEED}:{dataset_name}:stage0-true").sample(
        rejected_true,
        min(_STAGE0_SAMPLE_SIZE, len(rejected_true)),
    )
    examples_out = [
        {
            "dataset_row_index": index,
            "candidate": examples[index].candidate,
            "query": examples[index].query,
            "cosine_similarity": pair_scores[index].cosine_similarity,
            "gold_label": 1,
            "rejection_reasons": "|".join(rejection_reasons[index]),
        }
        for index in sampled
    ]
    return breakdown, examples_out


def _nli_score_rows(
    examples: Sequence[PairExample],
    pair_scores: Sequence[PairScore],
    stage0_reasons: Sequence[tuple[str, ...]],
    bidirectional_probabilities: Sequence[tuple[float, float] | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (example, pair_score, reasons, probabilities) in enumerate(
        zip(
            examples,
            pair_scores,
            stage0_reasons,
            bidirectional_probabilities,
            strict=True,
        )
    ):
        if (
            pair_score.cosine_similarity < min(_COSINE_THRESHOLDS)
            or reasons
            or probabilities is None
        ):
            continue
        rows.append(
            {
                "dataset_row_index": index,
                "gold_label": int(example.label),
                "cosine_similarity": pair_score.cosine_similarity,
                "candidate_to_query_entailment": probabilities[0],
                "query_to_candidate_entailment": probabilities[1],
                "bidirectional_min_entailment": min(probabilities),
            }
        )
    return rows


def _nli_histogram(
    dataset_name: str,
    examples: Sequence[PairExample],
    pair_scores: Sequence[PairScore],
    stage0_reasons: Sequence[tuple[str, ...]],
    bidirectional_probabilities: Sequence[tuple[float, float] | None],
) -> str:
    plot_path = (
        _RESULTS_DIRECTORY / f"paraphrase_ablation_{dataset_name}_nli_histograms.png"
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, (series_name, score_index) in zip(
        axes,
        (
            ("Candidate → query", 0),
            ("Query → candidate", 1),
            ("Bidirectional minimum", -1),
        ),
        strict=True,
    ):
        for label, gold_class in ((True, "Paraphrase"), (False, "Non-paraphrase")):
            values: list[float] = []
            for example, cosine, reasons, probabilities in zip(
                examples,
                pair_scores,
                stage0_reasons,
                bidirectional_probabilities,
                strict=True,
            ):
                if (
                    example.label == label
                    and cosine.cosine_similarity >= min(_COSINE_THRESHOLDS)
                    and not reasons
                    and probabilities is not None
                ):
                    values.append(
                        min(probabilities)
                        if score_index == -1
                        else probabilities[score_index]
                    )
            axis.hist(
                values,
                bins=20,
                range=(0, 1),
                alpha=0.55,
                label=f"{gold_class} (n={len(values)})",
            )
        axis.set(title=series_name, xlabel="Entailment probability")
        axis.grid(True, alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Pair count")
    figure.suptitle(
        f"{dataset_name.upper()} validation NLI scores "
        f"(cosine >= 0.85, Stage-0 passed)"
    )
    figure.tight_layout()
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return str(plot_path)


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fields),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows({**metadata, **row} for row in rows)


def _matched_precision_plot(
    dataset_name: str,
    matched_rows: Sequence[Mapping[str, object]],
    cosine_sweep_rows: Sequence[Mapping[str, object]],
    validation_sweep_rows: Sequence[Mapping[str, object]],
) -> str:
    plot_path = (
        _RESULTS_DIRECTORY
        / f"paraphrase_ablation_{dataset_name}_validation_matched_precision.png"
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    colors = {
        "cosine_only": "tab:blue",
        "cosine_stage0": "tab:orange",
        "cosine_stage0_nli": "tab:green",
    }
    markers = {0.80: "o", 0.90: "s", 0.95: "^"}
    curve_rows = [
        (
            "Cosine only",
            "tab:blue",
            "-",
            [
                row
                for row in cosine_sweep_rows
                if row["configuration"] == "cosine_only"
            ],
        ),
        (
            "Cosine + Stage-0",
            "tab:orange",
            "-",
            [
                row
                for row in cosine_sweep_rows
                if row["configuration"] == "cosine_stage0"
            ],
        ),
        *[
            (
                f"Cosine + Stage-0 + NLI (cosine {cosine_threshold:.2f})",
                "tab:green",
                "-" if cosine_threshold == 0.85 else "--",
                [
                    row
                    for row in validation_sweep_rows
                    if row["configuration"] == "cosine_stage0_nli"
                    and row["cosine_threshold"] == cosine_threshold
                ],
            )
            for cosine_threshold in _COSINE_THRESHOLDS
        ],
    ]
    for label, color, linestyle, rows in curve_rows:
        points = [
            (cast(float, row["precision"]), cast(float, row["recall"]))
            for row in rows
            if row.get("precision") is not None and row.get("recall") is not None
        ]
        points.sort(key=lambda point: point[0])
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                linestyle=linestyle,
                marker=".",
                markersize=3,
                label=label,
            )

    for row in matched_rows:
        if row.get("calibration_status") != "selected":
            continue
        precision = row.get("validation_precision")
        recall = row.get("validation_recall")
        if precision is None or recall is None:
            continue
        configuration = cast(str, row["configuration"])
        target = cast(float, row["precision_target"])
        axis.scatter(
            cast(float, precision),
            cast(float, recall),
            color=colors[configuration],
            marker=markers[target],
            edgecolors="black",
            linewidths=0.5,
            s=65,
            zorder=3,
        )
    target_legend = [
        Line2D(
            [],
            [],
            marker=markers[target],
            color="white",
            markeredgecolor="black",
            label=f"Train precision target {target:.2f}",
        )
        for target in _MATCHED_PRECISION_TARGETS
    ]
    axis.add_artist(axis.legend(handles=target_legend, loc="lower left", fontsize=8))
    axis.set(
        title=(
            f"{dataset_name.upper()} validation recall vs precision\n"
            "Train-calibrated operating points; marker shapes show precision target"
        ),
        xlabel="Validation precision",
        ylabel="Validation recall",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(True, alpha=0.25)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=8)
    else:
        axis.text(
            0.5,
            0.5,
            "No validation operating points were measured",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    figure.tight_layout()
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return str(plot_path)


def _save_outputs(
    dataset_name: str,
    *,
    repository: str,
    config: str,
    train_sample_count: int,
    validation_count: int,
    calibration_rows: list[dict[str, object]],
    selected_thresholds: list[dict[str, object]],
    selected_validation_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    cosine_sweep_rows: list[dict[str, object]],
    matched_precision_validation_rows: list[dict[str, object]],
    matched_precision_rows: list[dict[str, object]],
    stage0_breakdown: list[dict[str, object]],
    stage0_candidate_set_breakdown: list[dict[str, object]],
    stage0_samples: list[dict[str, object]],
    nli_scores: list[dict[str, object]],
    histogram_path: str,
    matched_precision_plot_path: str,
    embedding_model_source: str,
    run_date: str,
    git_commit: str,
    git_worktree_dirty: bool,
    cache_counts: Mapping[str, int],
) -> dict[str, str]:
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    prefix = f"paraphrase_ablation_{dataset_name}_train_calibrated_validation"
    calibration_path = _RESULTS_DIRECTORY / f"{prefix}_train_nli_sweep.csv"
    validation_path = _RESULTS_DIRECTORY / f"{prefix}_validation_nli_sweep.csv"
    selected_path = _RESULTS_DIRECTORY / f"{prefix}_selected_thresholds.csv"
    selected_validation_path = (
        _RESULTS_DIRECTORY / f"{prefix}_validation_selected_metrics.csv"
    )
    rejection_path = _RESULTS_DIRECTORY / f"{prefix}_stage0_rejections.csv"
    stage0_samples_path = _RESULTS_DIRECTORY / f"{prefix}_stage0_true_paraphrases.csv"
    nli_scores_path = _RESULTS_DIRECTORY / f"{prefix}_validation_nli_scores.csv"
    cosine_sweep_path = (
        _RESULTS_DIRECTORY / f"{prefix}_validation_cosine_sweep.csv"
    )
    matched_precision_path = (
        _RESULTS_DIRECTORY / f"{prefix}_validation_matched_precision.csv"
    )
    candidate_set_rejection_path = (
        _RESULTS_DIRECTORY / f"{prefix}_stage0_candidate_set_rejections.csv"
    )
    json_path = _RESULTS_DIRECTORY / f"{prefix}.json"

    common_metadata = {
        "dataset": dataset_name,
        "run_date_utc": run_date,
        "git_commit": git_commit,
        "embedding_model": _EMBEDDING_MODEL_NAME,
        "nli_model": _NLI_MODEL_NAME,
        "nli_model_revision": _NLI_MODEL_REVISION,
    }
    train_metadata = {**common_metadata, "split": "train"}
    validation_metadata = {**common_metadata, "split": "validation"}
    _write_csv(
        calibration_path,
        tuple(common_metadata) + ("split",)
        + (
            "cosine_threshold",
            "nli_threshold",
            "hits",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
        ),
        calibration_rows,
        train_metadata,
    )
    _write_csv(
        selected_validation_path,
        tuple(common_metadata) + ("split",)
        + (
            "cosine_threshold",
            "precision_target",
            "exists",
            "selected_nli_threshold",
            "hits",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
            "nli_pairs_sent",
            "nli_pairs_sent_percent",
        ),
        selected_validation_rows,
        validation_metadata,
    )
    _write_csv(
        validation_path,
        tuple(common_metadata) + ("split",)
        + (
            "cosine_threshold",
            "configuration",
            "nli_threshold",
            "hits",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
            "nli_pairs_sent",
            "nli_pairs_sent_percent",
        ),
        validation_rows,
        validation_metadata,
    )
    _write_csv(
        selected_path,
        tuple(common_metadata) + ("split",)
        + (
            "cosine_threshold",
            "precision_target",
            "exists",
            "selected_nli_threshold",
            "tie_break",
            "hits",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
        ),
        selected_thresholds,
        train_metadata,
    )
    _write_csv(
        rejection_path,
        tuple(common_metadata) + ("split", "candidate_set")
        + ("reason", "gold_label", "gold_class", "rejections", "counts_can_overlap"),
        stage0_breakdown,
        {**validation_metadata, "candidate_set": "all_validation"},
    )
    _write_csv(
        candidate_set_rejection_path,
        tuple(common_metadata)
        + (
            "split",
            "candidate_set",
            "cosine_threshold",
            "candidate_count",
            "reason",
            "gold_label",
            "gold_class",
            "rejections",
            "counts_can_overlap",
        ),
        stage0_candidate_set_breakdown,
        validation_metadata,
    )
    _write_csv(
        cosine_sweep_path,
        tuple(common_metadata)
        + (
            "split",
            "cosine_threshold",
            "configuration",
            "hits",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
        ),
        cosine_sweep_rows,
        validation_metadata,
    )
    _write_csv(
        matched_precision_path,
        tuple(common_metadata)
        + (
            "calibration_split",
            "split",
            "precision_target",
            "configuration",
            "calibration_status",
            "cosine_threshold",
            "nli_threshold",
            "train_hits",
            "train_precision",
            "train_recall",
            "validation_hits",
            "validation_true_positive",
            "validation_false_positive",
            "validation_false_negative",
            "validation_precision",
            "validation_recall",
            "validation_f1",
        ),
        matched_precision_rows,
        {**validation_metadata, "calibration_split": "train"},
    )
    _write_csv(
        stage0_samples_path,
        tuple(common_metadata) + ("split",)
        + (
            "dataset_row_index",
            "candidate",
            "query",
            "cosine_similarity",
            "gold_label",
            "rejection_reasons",
        ),
        stage0_samples,
        validation_metadata,
    )
    _write_csv(
        nli_scores_path,
        tuple(common_metadata)
        + (
            "split",
            "dataset_row_index",
            "gold_label",
            "cosine_similarity",
            "candidate_to_query_entailment",
            "query_to_candidate_entailment",
            "bidirectional_min_entailment",
        ),
        nli_scores,
        validation_metadata,
    )
    json_path.write_text(
        json.dumps(
            {
                **validation_metadata,
                "calibration_split": "train",
                "dataset_repository": repository,
                "configuration": config,
                "train_sample_count": train_sample_count,
                "train_sample_seed": _SAMPLE_SEED,
                "train_sampling": "deterministic reservoir sample",
                "validation_example_count": validation_count,
                "cosine_thresholds": list(_COSINE_THRESHOLDS),
                "nli_thresholds": list(_NLI_THRESHOLDS),
                "precision_targets": list(_PRECISION_TARGETS),
                "selected_thresholds": selected_thresholds,
                "validation_metrics_at_train_selected_thresholds": (
                    selected_validation_rows
                ),
                "validation_metrics": validation_rows,
                "cosine_sweep_thresholds": list(_COSINE_SWEEP_THRESHOLDS),
                "validation_cosine_sweep": cosine_sweep_rows,
                "validation_matched_precision_sweep": (
                    matched_precision_validation_rows
                ),
                "matched_precision_targets": list(_MATCHED_PRECISION_TARGETS),
                "validation_matched_precision": matched_precision_rows,
                "stage0_rejection_breakdown": stage0_breakdown,
                "stage0_candidate_set_rejection_breakdown": (
                    stage0_candidate_set_breakdown
                ),
                "stage0_behavior_changed_since_previous_run": False,
                "stage0_rejection_count_population": (
                    "The stage0_rejections CSV covers all validation examples; "
                    "the candidate-set CSV also reports counts for cosine>=each "
                    "swept threshold before Stage-0 filtering."
                ),
                "stage0_rejected_true_paraphrase_sample_count": len(stage0_samples),
                "nli_histogram_png": Path(histogram_path).name,
                "matched_precision_plot_png": Path(matched_precision_plot_path).name,
                "models": {
                    "embedding_model_source": embedding_model_source,
                    "fastembed_package_version": version("fastembed"),
                    "nli_model": _NLI_MODEL_NAME,
                    "nli_model_revision": _NLI_MODEL_REVISION,
                },
                "run_date_utc": run_date,
                "git_commit": git_commit,
                "git_worktree_dirty_at_run": git_worktree_dirty,
                "test_split_used": False,
                "score_cache_counts": dict(cache_counts),
                "notes": [
                    (
                        "NLI sweeps and selected operating points are calibrated "
                        "on train and evaluated on validation."
                    ),
                    (
                        "NLI score histograms include validation pairs with "
                        "cosine >= 0.85 that passed Stage-0."
                    ),
                    (
                        "Stage-0 number and negation rejection counts can overlap "
                        "when both reasons apply."
                    ),
                    (
                        "The original Stage-0 rejection breakdown covers the "
                        "entire validation split, not only cosine-retrieved pairs."
                    ),
                    (
                        "The expanded candidate-set breakdown labels all-validation "
                        "and cosine-threshold candidate populations separately."
                    ),
                    (
                        "Matched-precision calibration searches cosine thresholds "
                        "0.85-0.99 and NLI thresholds 0.01-0.99 at cosine 0.85 "
                        "and 0.95."
                    ),
                    (
                        "Stage-0 guard logic is behaviorally unchanged from the "
                        "previous evaluation; only its reason-reporting helper "
                        "was factored out."
                    ),
                ],
                "files": {
                    "train_nli_sweep_csv": calibration_path.name,
                    "validation_nli_sweep_csv": validation_path.name,
                    "selected_thresholds_csv": selected_path.name,
                    "validation_selected_metrics_csv": selected_validation_path.name,
                    "stage0_rejections_csv": rejection_path.name,
                    "stage0_candidate_set_rejections_csv": (
                        candidate_set_rejection_path.name
                    ),
                    "stage0_true_paraphrases_csv": stage0_samples_path.name,
                    "validation_nli_scores_csv": nli_scores_path.name,
                    "nli_histograms_png": Path(histogram_path).name,
                    "validation_cosine_sweep_csv": cosine_sweep_path.name,
                    "validation_matched_precision_csv": matched_precision_path.name,
                    "validation_matched_precision_plot_png": Path(
                        matched_precision_plot_path
                    ).name,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "train_nli_sweep_csv": str(calibration_path),
        "validation_nli_sweep_csv": str(validation_path),
        "selected_thresholds_csv": str(selected_path),
        "validation_selected_metrics_csv": str(selected_validation_path),
        "stage0_rejections_csv": str(rejection_path),
        "stage0_candidate_set_rejections_csv": str(candidate_set_rejection_path),
        "stage0_true_paraphrases_csv": str(stage0_samples_path),
        "validation_nli_scores_csv": str(nli_scores_path),
        "nli_histograms_png": histogram_path,
        "validation_cosine_sweep_csv": str(cosine_sweep_path),
        "validation_matched_precision_csv": str(matched_precision_path),
        "validation_matched_precision_plot_png": matched_precision_plot_path,
    }


async def _load_split_rows(
    repository: str,
    config: str,
    split: str,
    *,
    limit: int | None,
) -> list[Mapping[str, object]]:
    if split not in {"train", "validation"}:
        raise ValueError("only train and validation splits are permitted")
    dataset = load_dataset(repository, config, split=split, streaming=True)
    rows: Iterable[Mapping[str, object]] = cast(
        Iterable[Mapping[str, object]],
        dataset,
    )
    if split == "train":
        rows = _reservoir_sample(
            rows,
            min(_TRAIN_SAMPLE_SIZE, limit or _TRAIN_SAMPLE_SIZE),
            seed=_SAMPLE_SEED,
        )
    elif limit is not None:
        rows = itertools.islice(rows, limit)
    return list(rows)


async def run_dataset_ablation(
    dataset_name: str,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """Calibrate on a seeded train sample and report validation-only metrics."""
    if dataset_name not in _DATASETS:
        raise ValueError("dataset must be 'paws' or 'qqp'")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    repository, config, candidate_field, query_field = _DATASETS[dataset_name]
    run_date = datetime.now(UTC).isoformat()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_worktree_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    train_rows, validation_rows = await asyncio.gather(
        _load_split_rows(repository, config, "train", limit=limit),
        _load_split_rows(repository, config, "validation", limit=limit),
    )
    train_examples = _pair_examples(
        train_rows,
        candidate_field=candidate_field,
        query_field=query_field,
    )
    validation_examples = _pair_examples(
        validation_rows,
        candidate_field=candidate_field,
        query_field=query_field,
    )
    if not train_examples or not validation_examples:
        raise ValueError(f"{dataset_name} train/validation data must not be empty")

    score_cache_path = (
        _RESULTS_DIRECTORY / f"paraphrase_ablation_{dataset_name}_score_cache.jsonl"
    )
    score_cache = _load_score_cache(score_cache_path)
    prior_nli_count = _seed_cache_from_previous_samples(dataset_name, score_cache)
    embedder = FastEmbedder()
    scorer = OnnxNliScorer()
    try:
        train_scores, train_cosine_reused, train_cosine_computed = (
            await _ensure_cosine_scores(
                train_examples,
                embedder,
                score_cache,
                score_cache_path,
            )
        )
        validation_scores, validation_cosine_reused, validation_cosine_computed = (
            await _ensure_cosine_scores(
                validation_examples,
                embedder,
                score_cache,
                score_cache_path,
            )
        )
        train_reasons = [
            stage0_rejection_reasons(example.candidate, example.query)
            for example in train_examples
        ]
        validation_reasons = [
            stage0_rejection_reasons(example.candidate, example.query)
            for example in validation_examples
        ]
        train_probabilities, train_nli_reused, train_nli_computed = (
            await _ensure_nli_scores(
                train_examples,
                train_scores,
                train_reasons,
                embedder.model_name,
                scorer,
                score_cache,
                score_cache_path,
            )
        )
        validation_probabilities, validation_nli_reused, validation_nli_computed = (
            await _ensure_nli_scores(
                validation_examples,
                validation_scores,
                validation_reasons,
                embedder.model_name,
                scorer,
                score_cache,
                score_cache_path,
            )
        )
    finally:
        await scorer.close()
        await embedder.close()

    calibration_rows, selected_thresholds = _calibration_sweep(
        [example.label for example in train_examples],
        [score.cosine_similarity for score in train_scores],
        [not reasons for reasons in train_reasons],
        train_probabilities,
    )
    validation_metrics = _validation_sweep(
        [example.label for example in validation_examples],
        [score.cosine_similarity for score in validation_scores],
        [not reasons for reasons in validation_reasons],
        validation_probabilities,
    )
    train_labels = [example.label for example in train_examples]
    train_cosine_scores = [
        score.cosine_similarity for score in train_scores
    ]
    train_stage0_passes = [not reasons for reasons in train_reasons]
    validation_labels = [example.label for example in validation_examples]
    validation_cosine_scores = [
        score.cosine_similarity for score in validation_scores
    ]
    validation_stage0_passes = [not reasons for reasons in validation_reasons]
    cosine_sweep_rows = _cosine_validation_sweep(
        validation_labels,
        validation_cosine_scores,
        validation_stage0_passes,
    )
    matched_precision_rows = _matched_precision_rows(
        train_labels,
        train_cosine_scores,
        train_stage0_passes,
        train_probabilities,
        validation_labels,
        validation_cosine_scores,
        validation_stage0_passes,
        validation_probabilities,
    )
    matched_precision_plot_path = _matched_precision_plot(
        dataset_name,
        matched_precision_rows,
        cosine_sweep_rows,
        validation_metrics,
    )
    selected_validation_metrics = _selected_validation_rows(
        selected_thresholds,
        validation_metrics,
    )
    stage0_breakdown, stage0_samples = _stage0_analysis(
        validation_examples,
        validation_scores,
        validation_reasons,
        dataset_name=dataset_name,
    )
    stage0_candidate_set_breakdown = _stage0_candidate_set_analysis(
        validation_examples,
        validation_scores,
        validation_reasons,
    )
    validation_nli_scores = _nli_score_rows(
        validation_examples,
        validation_scores,
        validation_reasons,
        validation_probabilities,
    )
    histogram_path = _nli_histogram(
        dataset_name,
        validation_examples,
        validation_scores,
        validation_reasons,
        validation_probabilities,
    )
    model_info = next(
        (
            item
            for item in TextEmbedding.list_supported_models()
            if item.get("model") == embedder.model_name
        ),
        None,
    )
    if model_info is None:
        raise RuntimeError(f"FastEmbed model {embedder.model_name!r} is not registered")
    model_source = cast(dict[str, str], model_info["sources"]).get("hf", "unknown")
    cache_counts = {
        "prior_error_sample_nli_scores_loaded": prior_nli_count,
        "train_cosine_reused": train_cosine_reused,
        "train_cosine_computed": train_cosine_computed,
        "validation_cosine_reused": validation_cosine_reused,
        "validation_cosine_computed": validation_cosine_computed,
        "train_nli_reused": train_nli_reused,
        "train_nli_computed": train_nli_computed,
        "validation_nli_reused": validation_nli_reused,
        "validation_nli_computed": validation_nli_computed,
    }
    files = _save_outputs(
        dataset_name,
        repository=repository,
        config=config,
        train_sample_count=len(train_examples),
        validation_count=len(validation_examples),
        calibration_rows=calibration_rows,
        selected_thresholds=selected_thresholds,
        selected_validation_rows=selected_validation_metrics,
        validation_rows=validation_metrics,
        cosine_sweep_rows=cosine_sweep_rows,
        matched_precision_validation_rows=validation_metrics,
        matched_precision_rows=matched_precision_rows,
        stage0_breakdown=stage0_breakdown,
        stage0_candidate_set_breakdown=stage0_candidate_set_breakdown,
        stage0_samples=stage0_samples,
        nli_scores=validation_nli_scores,
        histogram_path=histogram_path,
        matched_precision_plot_path=matched_precision_plot_path,
        embedding_model_source=model_source,
        run_date=run_date,
        git_commit=git_commit,
        git_worktree_dirty=git_worktree_dirty,
        cache_counts=cache_counts,
    )
    return {
        "dataset": dataset_name,
        "train_sample_count": len(train_examples),
        "validation_count": len(validation_examples),
        "selected_thresholds": selected_thresholds,
        "validation_metrics": validation_metrics,
        "validation_cosine_sweep": cosine_sweep_rows,
        "validation_matched_precision": matched_precision_rows,
        "stage0_rejection_breakdown": stage0_breakdown,
        "stage0_candidate_set_rejection_breakdown": stage0_candidate_set_breakdown,
        "stage0_behavior_changed_since_previous_run": False,
        "stage0_rejected_true_paraphrase_sample_count": len(stage0_samples),
        "score_cache_counts": cache_counts,
        "files": files,
        "test_split_used": False,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate NLI on train samples; evaluate PAWS and QQP validation."
    )
    parser.add_argument("--dataset", choices=("paws", "qqp", "all"), default="all")
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke-test limit for train reservoir and validation rows.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    datasets = ("paws", "qqp") if args.dataset == "all" else (args.dataset,)
    results: list[dict[str, object]] = []
    for dataset_name in datasets:
        results.append(await run_dataset_ablation(dataset_name, limit=args.limit))
    return results


def main() -> None:
    """Run train-calibrated validation reports and print JSON summaries."""
    reports = asyncio.run(_run(_arguments()))
    summaries = [
        {
            "dataset": report["dataset"],
            "train_sample_count": report["train_sample_count"],
            "validation_count": report["validation_count"],
            "selected_thresholds": report["selected_thresholds"],
            "stage0_rejected_true_paraphrase_sample_count": (
                report["stage0_rejected_true_paraphrase_sample_count"]
            ),
            "score_cache_counts": report["score_cache_counts"],
            "files": report["files"],
            "test_split_used": report["test_split_used"],
        }
        for report in reports
    ]
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
