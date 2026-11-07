"""Validation-only ablation of cosine, Stage-0, and bidirectional NLI."""

import argparse
import asyncio
import csv
import json
import random
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import cast

from datasets import load_dataset  # type: ignore[import-untyped]
from fastembed import TextEmbedding

from aegis.engine.embeddings import FastEmbedder
from aegis.engine.nli_reranker import OnnxNliScorer
from aegis.engine.regex_guard import stage0_guard
from benchmarks.evaluation import (
    PairExample,
    PairScore,
    binary_metrics,
    rows_to_pair_examples,
    score_pair_cosines,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIRECTORY = _REPOSITORY_ROOT / "benchmarks" / "results"
_COSINE_THRESHOLDS = (0.85, 0.88, 0.90, 0.92, 0.95)
_NLI_THRESHOLDS = tuple(value / 100 for value in range(101))
_EXAMPLES_PER_BATCH = 256
_NLI_PAIRS_PER_BATCH = 128
_FALSE_POSITIVE_SAMPLE_SIZE = 15
_SAMPLE_SEED = 42
_NLI_MODEL_NAME = "Xenova/nli-deberta-v3-small"
_NLI_MODEL_REVISION = "6bc2a55c7c0f7e2bc68de60bb248e523e2612abb"

_DATASETS: dict[str, tuple[str, str, str, str]] = {
    "paws": (
        "google-research-datasets/paws",
        "labeled_final",
        "sentence1",
        "sentence2",
    ),
    "qqp": ("nyu-mll/glue", "qqp", "question1", "question2"),
}


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


def select_nli_threshold(
    labels: Sequence[bool],
    cosine_scores: Sequence[float],
    stage0_passes: Sequence[bool],
    bidirectional_probabilities: Sequence[tuple[float, float] | None],
    *,
    cosine_thresholds: Sequence[float] = _COSINE_THRESHOLDS,
    nli_thresholds: Sequence[float] = _NLI_THRESHOLDS,
) -> tuple[float, list[dict[str, float]]]:
    """Choose one NLI threshold maximizing mean F1 over requested cosine points.

    Ties select the higher NLI threshold as the more conservative option.
    """
    lengths = {
        len(labels),
        len(cosine_scores),
        len(stage0_passes),
        len(bidirectional_probabilities),
    }
    if len(lengths) != 1:
        raise ValueError("calibration inputs must have equal length")
    if not cosine_thresholds or not nli_thresholds:
        raise ValueError("threshold collections must not be empty")

    calibration: list[dict[str, float]] = []
    for nli_threshold in nli_thresholds:
        f1_values: list[float] = []
        for cosine_threshold in cosine_thresholds:
            predictions = [
                similarity >= cosine_threshold
                and stage0_pass
                and probabilities is not None
                and probabilities[0] >= nli_threshold
                and probabilities[1] >= nli_threshold
                for similarity, stage0_pass, probabilities in zip(
                    cosine_scores,
                    stage0_passes,
                    bidirectional_probabilities,
                    strict=True,
                )
            ]
            result = _metrics(labels, predictions)
            f1_values.append(cast(float, result["f1"] or 0.0))
        calibration.append(
            {
                "nli_threshold": nli_threshold,
                "mean_f1": sum(f1_values) / len(f1_values),
            }
        )
    selected = max(
        calibration,
        key=lambda item: (item["mean_f1"], item["nli_threshold"]),
    )
    return selected["nli_threshold"], calibration


def _ablation_rows(
    examples: Sequence[PairExample],
    pair_scores: Sequence[PairScore],
    stage0_passes: Sequence[bool],
    bidirectional_probabilities: Sequence[tuple[float, float] | None],
    *,
    nli_threshold: float,
    dataset_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    labels = [example.label for example in examples]
    similarities = [score.cosine_similarity for score in pair_scores]
    metrics_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    configurations = ("cosine_only", "cosine_stage0", "cosine_stage0_nli")

    for cosine_threshold in _COSINE_THRESHOLDS:
        predictions_by_configuration: dict[str, list[bool]] = {
            "cosine_only": [],
            "cosine_stage0": [],
            "cosine_stage0_nli": [],
        }
        nli_sent_indices: list[int] = []
        for index, (similarity, stage0_pass, probabilities) in enumerate(
            zip(
                similarities,
                stage0_passes,
                bidirectional_probabilities,
                strict=True,
            )
        ):
            cosine_pass = similarity >= cosine_threshold
            predictions_by_configuration["cosine_only"].append(cosine_pass)
            predictions_by_configuration["cosine_stage0"].append(
                cosine_pass and stage0_pass
            )
            sent_to_nli = cosine_pass and stage0_pass
            if sent_to_nli:
                nli_sent_indices.append(index)
            predictions_by_configuration["cosine_stage0_nli"].append(
                sent_to_nli
                and probabilities is not None
                and probabilities[0] >= nli_threshold
                and probabilities[1] >= nli_threshold
            )

        for configuration in configurations:
            predictions = predictions_by_configuration[configuration]
            row: dict[str, object] = {
                "cosine_threshold": cosine_threshold,
                "configuration": configuration,
                "nli_threshold": (
                    nli_threshold if configuration == "cosine_stage0_nli" else None
                ),
                **_metrics(labels, predictions),
                "nli_pairs_sent": (
                    len(nli_sent_indices)
                    if configuration == "cosine_stage0_nli"
                    else None
                ),
                "nli_pairs_sent_percent": (
                    len(nli_sent_indices) / len(examples) * 100
                    if configuration == "cosine_stage0_nli" and examples
                    else (0.0 if configuration == "cosine_stage0_nli" else None)
                ),
            }
            metrics_rows.append(row)

            false_positive_indices = [
                index
                for index, (label, prediction) in enumerate(
                    zip(labels, predictions, strict=True)
                )
                if prediction and not label
            ]
            sampled_indices = random.Random(
                f"{_SAMPLE_SEED}:{dataset_name}:{cosine_threshold}:{configuration}"
            ).sample(
                false_positive_indices,
                min(_FALSE_POSITIVE_SAMPLE_SIZE, len(false_positive_indices)),
            )
            for index in sampled_indices:
                score = pair_scores[index]
                probabilities = bidirectional_probabilities[index]
                error_rows.append(
                    {
                        "cosine_threshold": cosine_threshold,
                        "configuration": configuration,
                        "nli_threshold": (
                            nli_threshold
                            if configuration == "cosine_stage0_nli"
                            else None
                        ),
                        "dataset_row_index": index,
                        "candidate": score.example.candidate,
                        "query": score.example.query,
                        "cosine_similarity": score.cosine_similarity,
                        "gold_label": int(score.example.label),
                        "candidate_to_query_entailment": (
                            probabilities[0] if probabilities is not None else None
                        ),
                        "query_to_candidate_entailment": (
                            probabilities[1] if probabilities is not None else None
                        ),
                    }
                )
    return metrics_rows, error_rows


async def _embed_all_pairs(
    examples: Sequence[PairExample],
    embedder: FastEmbedder,
) -> list[PairScore]:
    pair_scores: list[PairScore] = []
    for offset in range(0, len(examples), _EXAMPLES_PER_BATCH):
        batch_scores, _, _ = await score_pair_cosines(
            examples[offset : offset + _EXAMPLES_PER_BATCH],
            embedder,
        )
        pair_scores.extend(batch_scores)
    return pair_scores


async def _score_nli_candidates(
    examples: Sequence[PairExample],
    pair_scores: Sequence[PairScore],
    stage0_passes: Sequence[bool],
    scorer: OnnxNliScorer,
) -> list[tuple[float, float] | None]:
    """Score each Stage-0-safe pair at the minimum cosine operating point."""
    candidate_indices = [
        index
        for index, (score, stage0_pass) in enumerate(
            zip(pair_scores, stage0_passes, strict=True)
        )
        if score.cosine_similarity >= min(_COSINE_THRESHOLDS) and stage0_pass
    ]
    probabilities: list[tuple[float, float] | None] = [None] * len(examples)
    for offset in range(0, len(candidate_indices), _NLI_PAIRS_PER_BATCH):
        batch_indices = candidate_indices[offset : offset + _NLI_PAIRS_PER_BATCH]
        pairs = [
            pair
            for index in batch_indices
            for pair in (
                (examples[index].candidate, examples[index].query),
                (examples[index].query, examples[index].candidate),
            )
        ]
        scores = await scorer.entailment_probabilities(pairs)
        if len(scores) != len(pairs):
            raise RuntimeError(
                f"NLI scorer returned {len(scores)} scores; expected {len(pairs)}"
            )
        if any(not 0 <= score <= 1 for score in scores):
            raise RuntimeError("NLI scorer returned an invalid probability")
        for item_index, dataset_index in enumerate(batch_indices):
            probabilities[dataset_index] = (
                scores[item_index * 2],
                scores[item_index * 2 + 1],
            )
    return probabilities


def _save_outputs(
    dataset_name: str,
    *,
    repository: str,
    config: str,
    split: str,
    example_count: int,
    nli_threshold: float,
    calibration: list[dict[str, float]],
    metrics_rows: list[dict[str, object]],
    error_rows: list[dict[str, object]],
    embedding_model_source: str,
    run_date: str,
    git_commit: str,
    git_worktree_dirty: bool,
    limit: int | None,
) -> dict[str, str]:
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    prefix = f"paraphrase_ablation_{dataset_name}_validation"
    metrics_path = _RESULTS_DIRECTORY / f"{prefix}_metrics.csv"
    errors_path = _RESULTS_DIRECTORY / f"{prefix}_false_positives.csv"
    json_path = _RESULTS_DIRECTORY / f"{prefix}.json"

    csv_metadata = {
        "dataset": dataset_name,
        "split": split,
        "run_date_utc": run_date,
        "git_commit": git_commit,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "nli_model": _NLI_MODEL_NAME,
        "nli_model_revision": _NLI_MODEL_REVISION,
    }
    metadata_fields = list(csv_metadata)
    with metrics_path.open("w", encoding="utf-8", newline="") as output_file:
        fields = metadata_fields + [
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
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**csv_metadata, **row} for row in metrics_rows)

    with errors_path.open("w", encoding="utf-8", newline="") as output_file:
        fields = metadata_fields + [
            "cosine_threshold",
            "configuration",
            "nli_threshold",
            "dataset_row_index",
            "candidate",
            "query",
            "cosine_similarity",
            "gold_label",
            "candidate_to_query_entailment",
            "query_to_candidate_entailment",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**csv_metadata, **row} for row in error_rows)

    metadata = {
        "dataset": dataset_name,
        "dataset_repository": repository,
        "configuration": config,
        "split": split,
        "example_count": example_count,
        "limit": limit,
        "label_meaning": "1=paraphrase, 0=not paraphrase",
        "cosine_thresholds": list(_COSINE_THRESHOLDS),
        "configurations": [
            "cosine_only",
            "cosine_stage0",
            "cosine_stage0_nli",
        ],
        "nli_threshold_selection": {
            "selected_threshold": nli_threshold,
            "criterion": (
                "maximum mean F1 across the five requested cosine thresholds "
                "on this validation split; ties select the higher threshold"
            ),
            "calibration_curve": calibration,
        },
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_model_source": embedding_model_source,
        "fastembed_package_version": version("fastembed"),
        "nli_model": _NLI_MODEL_NAME,
        "nli_model_revision": _NLI_MODEL_REVISION,
        "run_date_utc": run_date,
        "git_commit": git_commit,
        "git_worktree_dirty_at_run": git_worktree_dirty,
        "test_split_used": False,
        "metrics": metrics_rows,
        "sampled_false_positives": error_rows,
        "files": {
            "metrics_csv": metrics_path.name,
            "false_positives_csv": errors_path.name,
        },
    }
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "metrics_csv": str(metrics_path),
        "false_positives_csv": str(errors_path),
    }


async def run_dataset_ablation(
    dataset_name: str,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """Run one dataset's ablation using only its validation split."""
    if dataset_name not in _DATASETS:
        raise ValueError("dataset must be 'paws' or 'qqp'")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    repository, config, candidate_field, query_field = _DATASETS[dataset_name]
    split = "validation"
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
    dataset = load_dataset(repository, config, split=split)
    examples = rows_to_pair_examples(
        cast(Iterable[Mapping[str, object]], dataset),
        text_a_field=candidate_field,
        text_b_field=query_field,
        label_field="label",
        limit=limit,
    )
    if not examples:
        raise ValueError(f"{dataset_name} validation split contains no examples")

    embedder = FastEmbedder()
    scorer = OnnxNliScorer()
    try:
        pair_scores = await _embed_all_pairs(examples, embedder)
        stage0_passes = [
            stage0_guard(example.candidate, example.query) for example in examples
        ]
        bidirectional_probabilities = await _score_nli_candidates(
            examples,
            pair_scores,
            stage0_passes,
            scorer,
        )
    finally:
        await scorer.close()
        await embedder.close()

    labels = [example.label for example in examples]
    similarities = [score.cosine_similarity for score in pair_scores]
    nli_threshold, calibration = select_nli_threshold(
        labels,
        similarities,
        stage0_passes,
        bidirectional_probabilities,
    )
    metrics_rows, error_rows = _ablation_rows(
        examples,
        pair_scores,
        stage0_passes,
        bidirectional_probabilities,
        nli_threshold=nli_threshold,
        dataset_name=dataset_name,
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
    files = _save_outputs(
        dataset_name,
        repository=repository,
        config=config,
        split=split,
        example_count=len(examples),
        nli_threshold=nli_threshold,
        calibration=calibration,
        metrics_rows=metrics_rows,
        error_rows=error_rows,
        embedding_model_source=model_source,
        run_date=run_date,
        git_commit=git_commit,
        git_worktree_dirty=git_worktree_dirty,
        limit=limit,
    )
    return {
        "dataset": dataset_name,
        "split": split,
        "example_count": len(examples),
        "nli_threshold": nli_threshold,
        "metrics": metrics_rows,
        "sampled_false_positive_count": len(error_rows),
        "metadata": {
            "embedding_model": embedder.model_name,
            "embedding_model_source": model_source,
            "fastembed_package_version": version("fastembed"),
            "nli_model": _NLI_MODEL_NAME,
            "nli_model_revision": _NLI_MODEL_REVISION,
            "run_date_utc": run_date,
            "git_commit": git_commit,
            "test_split_used": False,
        },
        "files": files,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-only PAWS and QQP ablation reports."
    )
    parser.add_argument("--dataset", choices=("paws", "qqp", "all"), default="all")
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N validation examples per dataset.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    datasets = ("paws", "qqp") if args.dataset == "all" else (args.dataset,)
    return [
        await run_dataset_ablation(dataset_name, limit=args.limit)
        for dataset_name in datasets
    ]


def main() -> None:
    """Run requested validation ablations and print results as JSON."""
    print(json.dumps(asyncio.run(_run(_arguments())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
