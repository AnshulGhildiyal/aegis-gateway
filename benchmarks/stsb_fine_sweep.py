"""Fine-grained STS-B validation sweep and error analysis artifacts."""

import asyncio
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from datasets import load_dataset  # type: ignore[import-untyped]
from fastembed import TextEmbedding

from aegis.engine.embeddings import FastEmbedder
from benchmarks.evaluation import (
    PairExample,
    PairScore,
    score_pair_cosines,
    stsb_binary_labels,
    threshold_sweep,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIRECTORY = _REPOSITORY_ROOT / "benchmarks" / "results"
_DATASET = "nyu-mll/glue"
_CONFIG = "stsb"
_SPLIT = "validation"
_THRESHOLDS = [value / 100 for value in range(80, 100)]


@dataclass(frozen=True, slots=True)
class ThresholdSummary:
    """Operating-point metrics at one selected threshold."""

    name: str
    threshold: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    hit_count: int | None
    false_positive_count: int | None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _summaries(
    results: list[dict[str, int | float | None]],
) -> list[ThresholdSummary]:
    enriched: list[tuple[dict[str, int | float | None], float | None]] = []
    for row in results:
        precision = cast(float | None, row["precision"])
        recall = cast(float | None, row["recall"])
        enriched.append((row, _f1(precision, recall)))

    best_row, best_f1 = max(
        enriched,
        key=lambda item: (
            item[1] if item[1] is not None else -math.inf,
            -cast(float, item[0]["threshold"]),
        ),
    )
    targets: list[tuple[str, float | None, dict[str, int | float | None] | None]] = [
        ("best_f1", best_f1, best_row),
    ]
    for target_precision in (0.95, 0.90):
        matching = [
            row
            for row, _ in enriched
            if cast(float | None, row["precision"]) is not None
            and cast(float, row["precision"]) >= target_precision
        ]
        selected = (
            min(matching, key=lambda row: cast(float, row["threshold"]))
            if matching
            else None
        )
        name = f"lowest_threshold_precision_at_least_{target_precision:.2f}"
        targets.append(
            (
                name,
                _f1(
                    cast(float | None, selected["precision"]),
                    cast(float | None, selected["recall"]),
                )
                if selected is not None
                else None,
                selected,
            )
        )

    summaries: list[ThresholdSummary] = []
    for name, f1_value, selected_row in targets:
        if selected_row is None:
            summaries.append(ThresholdSummary(name, None, None, None, None, None, None))
            continue
        summaries.append(
            ThresholdSummary(
                name=name,
                threshold=cast(float, selected_row["threshold"]),
                precision=cast(float | None, selected_row["precision"]),
                recall=cast(float | None, selected_row["recall"]),
                f1=f1_value,
                hit_count=cast(int, selected_row["true_positive"])
                + cast(int, selected_row["false_positive"]),
                false_positive_count=cast(int, selected_row["false_positive"]),
            )
        )
    return summaries


def _error_rows(
    scores: list[PairScore],
    labels: list[bool],
    gold_scores: list[float],
) -> list[dict[str, str | float | int]]:
    error_groups = (
        ("false_positive_at_0.92", 0.92, False, 15),
        ("false_positive_at_0.96", 0.96, False, None),
        ("false_negative_at_0.88", 0.88, True, 10),
    )
    rows: list[dict[str, str | float | int]] = []
    for group_name, threshold, gold_label, limit in error_groups:
        errors = [
            (score, label, gold_score)
            for score, label, gold_score in zip(
                scores,
                labels,
                gold_scores,
                strict=True,
            )
            if label == gold_label
            and (
                score.cosine_similarity < threshold
                if gold_label
                else score.cosine_similarity >= threshold
            )
        ]
        errors.sort(
            key=lambda item: item[0].cosine_similarity,
            reverse=True,
        )
        if limit is not None:
            errors = errors[:limit]
        for score, label, gold_score in errors:
            rows.append(
                {
                    "error_group": group_name,
                    "candidate_sentence": score.example.candidate,
                    "query_sentence": score.example.query,
                    "cosine_score": score.cosine_similarity,
                    "gold_score": gold_score,
                    "gold_label": int(label),
                    "threshold": threshold,
                }
            )
    return rows


def _save_results(
    scores: list[PairScore],
    labels: list[bool],
    results: list[dict[str, int | float | None]],
    summaries: list[ThresholdSummary],
    errors: list[dict[str, str | float | int]],
    *,
    model_name: str,
    embedding_model_source: str,
    run_date: str,
    git_commit: str,
    git_worktree_dirty: bool,
    embedding_seconds: float,
    cosine_seconds: list[float],
) -> dict[str, str]:
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    metrics_path = _RESULTS_DIRECTORY / "sts_b_validation_fine_sweep.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as output_file:
        fields = [
            "threshold",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "accuracy",
            "hit_count",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            precision = cast(float | None, result["precision"])
            recall = cast(float | None, result["recall"])
            writer.writerow(
                {
                    **result,
                    "f1": _f1(precision, recall),
                    "hit_count": cast(int, result["true_positive"])
                    + cast(int, result["false_positive"]),
                }
            )

    error_path = _RESULTS_DIRECTORY / "sts_b_validation_error_analysis.csv"
    with error_path.open("w", encoding="utf-8", newline="") as output_file:
        fields = [
            "error_group",
            "candidate_sentence",
            "query_sentence",
            "cosine_score",
            "gold_score",
            "gold_label",
            "threshold",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    thresholds = [cast(float, result["threshold"]) for result in results]
    precisions = [cast(float, result["precision"]) for result in results]
    recalls = [cast(float, result["recall"]) for result in results]
    f1_values = [
        cast(float, _f1(precision, recall))
        for precision, recall in zip(precisions, recalls, strict=True)
    ]

    threshold_plot = _RESULTS_DIRECTORY / "sts_b_validation_metrics_vs_threshold.png"
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(thresholds, precisions, marker=".", label="Precision")
    axis.plot(thresholds, recalls, marker=".", label="Recall")
    axis.plot(thresholds, f1_values, marker=".", label="F1")
    axis.set(
        xlabel="Cosine threshold",
        ylabel="Score",
        title="STS-B validation: precision, recall, and F1",
        ylim=(0, 1.02),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(threshold_plot, dpi=160)
    plt.close(figure)

    pr_plot = _RESULTS_DIRECTORY / "sts_b_validation_precision_recall.png"
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recalls, precisions, marker=".")
    for threshold, precision, recall in zip(
        thresholds,
        precisions,
        recalls,
        strict=True,
    ):
        if threshold in {0.80, 0.85, 0.90, 0.95, 0.99}:
            axis.annotate(f"{threshold:.2f}", (recall, precision))
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="STS-B validation: precision-recall curve",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(pr_plot, dpi=160)
    plt.close(figure)

    metadata_path = _RESULTS_DIRECTORY / "sts_b_validation_metadata.json"
    metadata = {
        "dataset": _DATASET,
        "configuration": _CONFIG,
        "split": _SPLIT,
        "example_count": len(scores),
        "positive_count_score_gte_4": sum(labels),
        "thresholds": _THRESHOLDS,
        "embedding_model": model_name,
        "embedding_model_source": embedding_model_source,
        "fastembed_package_version": version("fastembed"),
        "run_date_utc": run_date,
        "git_commit": git_commit,
        "git_worktree_dirty_at_run": git_worktree_dirty,
        "embedding_batch_seconds": embedding_seconds,
        "cosine_scoring_mean_ms_per_pair": (
            float(np.mean(cosine_seconds) * 1000) if cosine_seconds else None
        ),
        "selected_thresholds": [asdict(summary) for summary in summaries],
        "files": {
            "metrics_csv": metrics_path.name,
            "error_analysis_csv": error_path.name,
            "metrics_plot": threshold_plot.name,
            "precision_recall_plot": pr_plot.name,
        },
        "test_split_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics_csv": str(metrics_path),
        "error_analysis_csv": str(error_path),
        "metrics_plot": str(threshold_plot),
        "precision_recall_plot": str(pr_plot),
        "metadata_json": str(metadata_path),
    }


async def run_stsb_validation_sweep() -> dict[str, object]:
    """Load only STS-B validation, calculate metrics, and save reproducible outputs."""
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
    dataset = load_dataset(_DATASET, _CONFIG, split=_SPLIT)
    raw_scores = [float(row["label"]) for row in dataset]
    labels = stsb_binary_labels(raw_scores)
    examples = [
        PairExample(
            candidate=str(row["sentence1"]),
            query=str(row["sentence2"]),
            label=label,
        )
        for row, label in zip(dataset, labels, strict=True)
    ]

    embedder = FastEmbedder()
    try:
        pair_scores, embedding_seconds, cosine_seconds = await score_pair_cosines(
            examples,
            embedder,
        )
    finally:
        await embedder.close()

    similarities = [score.cosine_similarity for score in pair_scores]
    results = threshold_sweep(labels, similarities, _THRESHOLDS)
    summaries = _summaries(results)
    errors = _error_rows(pair_scores, labels, raw_scores)
    registry_model = next(
        (
            model
            for model in TextEmbedding.list_supported_models()
            if model.get("model") == embedder.model_name
        ),
        None,
    )
    if registry_model is None:
        raise RuntimeError(f"FastEmbed model {embedder.model_name!r} is not registered")
    model_source = cast(dict[str, str], registry_model["sources"]).get("hf", "unknown")
    files = _save_results(
        pair_scores,
        labels,
        results,
        summaries,
        errors,
        model_name=embedder.model_name,
        embedding_model_source=model_source,
        run_date=run_date,
        git_commit=git_commit,
        git_worktree_dirty=git_worktree_dirty,
        embedding_seconds=embedding_seconds,
        cosine_seconds=cosine_seconds,
    )
    return {
        "dataset": f"{_DATASET}/{_CONFIG}",
        "split": _SPLIT,
        "example_count": len(pair_scores),
        "positive_count": sum(labels),
        "threshold_results": results,
        "selected_thresholds": summaries,
        "error_analysis": errors,
        "metadata": {
            "embedding_model": embedder.model_name,
            "embedding_model_source": model_source,
            "fastembed_package_version": version("fastembed"),
            "run_date_utc": run_date,
            "git_commit": git_commit,
        },
        "files": files,
        "test_split_used": False,
    }


def main() -> None:
    """Run validation-only sweep and print metrics and selected error examples."""
    result = asyncio.run(run_stsb_validation_sweep())
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
