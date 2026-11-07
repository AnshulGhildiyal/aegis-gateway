"""Banking77 leave-one-out train calibration and frozen test evaluation."""

import asyncio
import csv
import hashlib
import json
import math
import random
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D

from aegis.engine.embeddings import FastEmbedder
from aegis.engine.nli_reranker import OnnxNliScorer
from aegis.engine.regex_guard import stage0_guard
from aegis.storage.cache import CacheKey
from benchmarks.evaluate_banking77 import _read_splits
from benchmarks.evaluation import IntentExample

_RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"
_COSINE_THRESHOLDS = tuple(value / 100 for value in range(85, 100))
_NLI_COSINE_THRESHOLDS = (0.85, 0.90, 0.95)
_NLI_THRESHOLDS = tuple(value / 100 for value in range(1, 100))
_PRECISION_TARGETS = (0.95, 0.97, 0.98)
_TOP_CANDIDATES = 2
_NLI_PAIRS_PER_BATCH = 64
_FP_AUDIT_SIZE = 40
_RANDOM_SEED = 42
_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_NLI_MODEL = "Xenova/nli-deberta-v3-small"
_NLI_REVISION = "6bc2a55c7c0f7e2bc68de60bb248e523e2612abb"
_CACHE_KEY = CacheKey(
    system_prompt_hash="banking77-loo-evaluation",
    tenant_id="banking77-loo-evaluation",
    embedding_model_version=_EMBEDDING_MODEL,
    target_llm="banking77-loo-evaluation",
    temperature=0.0,
    max_tokens=None,
)

Candidate = tuple[int, float]
CandidateRows = list[list[Candidate]]
NliPair = tuple[str, str]
NliScores = dict[str, tuple[float, float]]


def _arguments() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Calibrate Banking77 thresholds with train leave-one-out and "
            "evaluate only the selected thresholds on test."
        )
    )
    parser.parse_args()


def _pair_key(candidate: str, query: str) -> str:
    payload = json.dumps(
        [_NLI_MODEL, _NLI_REVISION, candidate, query],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _vector_cache_path(split: str, model_name: str) -> Path:
    model_key = hashlib.sha256(
        f"{model_name}:{version('fastembed')}".encode()
    ).hexdigest()[:16]
    return _RESULTS_DIRECTORY / f"banking77_{split}_{model_key}_embeddings.npz"


async def _cached_embeddings(
    examples: Sequence[IntentExample],
    *,
    split: str,
    embedder: FastEmbedder,
) -> tuple[np.ndarray, bool]:
    path = _vector_cache_path(split, embedder.model_name)
    text_keys = np.asarray(
        [
            hashlib.sha256(example.text.encode("utf-8")).hexdigest()
            for example in examples
        ]
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            cached_model = str(archive["model_name"].item())
            cached_text_keys = archive["text_keys"]
            vectors = archive["vectors"]
        if cached_model == embedder.model_name and np.array_equal(
            cached_text_keys, text_keys
        ):
            if (
                vectors.ndim != 2
                or vectors.shape[0] != len(examples)
                or not np.isfinite(vectors).all()
            ):
                raise ValueError(f"invalid embedding cache at {path}")
            return _normalize_vectors(vectors), True

    vectors = _normalize_vectors(
        np.asarray(
            await embedder.embed([example.text for example in examples]),
            dtype=np.float32,
        )
    )
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez_compressed(
            output,
            model_name=np.asarray(embedder.model_name),
            text_keys=text_keys,
            vectors=vectors,
        )
    return vectors, False


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise RuntimeError("embedder returned invalid vectors")
    norms = np.linalg.norm(vectors, axis=1)
    normalized = np.divide(
        vectors,
        norms[:, np.newaxis],
        out=np.zeros_like(vectors, dtype=np.float32),
        where=norms[:, np.newaxis] != 0,
    )
    return np.asarray(normalized, dtype=np.float32)


def _top_candidates(
    query_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    leave_one_out: bool,
) -> CandidateRows:
    if query_vectors.ndim != 2 or candidate_vectors.ndim != 2:
        raise ValueError("query and candidate vectors must be matrices")
    if query_vectors.shape[1] != candidate_vectors.shape[1]:
        raise ValueError("query and candidate embedding dimensions must match")
    if leave_one_out and query_vectors.shape[0] != candidate_vectors.shape[0]:
        raise ValueError("leave-one-out requires matching query and candidate rows")
    if candidate_vectors.shape[0] < 2 and leave_one_out:
        return [[] for _ in range(query_vectors.shape[0])]

    output: CandidateRows = []
    block_size = 128
    for start in range(0, len(query_vectors), block_size):
        stop = min(start + block_size, len(query_vectors))
        similarities = query_vectors[start:stop] @ candidate_vectors.T
        similarities = np.clip(similarities, -1.0, 1.0)
        if leave_one_out:
            row_indices = np.arange(start, stop)
            similarities[np.arange(stop - start), row_indices] = -np.inf
        candidate_count = min(_TOP_CANDIDATES, candidate_vectors.shape[0])
        for _offset, row in enumerate(similarities):
            indices = np.argpartition(row, -candidate_count)[-candidate_count:]
            ordered = sorted(
                indices.tolist(),
                key=lambda index: (-float(row[index]), index),
            )
            output.append(
                [(int(index), float(row[index])) for index in ordered]
            )
    return output


def _load_nli_cache(path: Path) -> NliScores:
    scores: NliScores = {}
    if not path.exists():
        return scores
    with path.open(encoding="utf-8") as cache_file:
        for line_number, line in enumerate(cache_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"invalid NLI cache record at {path}:{line_number}")
            key = record.get("key")
            forward = record.get("candidate_to_query")
            reverse = record.get("query_to_candidate")
            if (
                not isinstance(key, str)
                or not isinstance(forward, (int, float))
                or not isinstance(reverse, (int, float))
                or not math.isfinite(forward)
                or not math.isfinite(reverse)
                or not 0 <= forward <= 1
                or not 0 <= reverse <= 1
            ):
                raise ValueError(f"invalid NLI cache record at {path}:{line_number}")
            scores[key] = (float(forward), float(reverse))
    return scores


async def _ensure_nli_scores(
    queries: Sequence[IntentExample],
    candidate_examples: Sequence[IntentExample],
    candidate_rows: CandidateRows,
    *,
    scorer: OnnxNliScorer,
    score_cache: NliScores,
    cache_path: Path,
) -> tuple[int, int]:
    pairs: dict[str, NliPair] = {}
    for query_index, candidate_row in enumerate(candidate_rows):
        query = queries[query_index]
        for candidate_index, similarity in candidate_row:
            if similarity < _NLI_COSINE_THRESHOLDS[0]:
                continue
            candidate = candidate_examples[candidate_index]
            if not stage0_guard(candidate.text, query.text):
                continue
            key = _pair_key(candidate.text, query.text)
            pairs.setdefault(key, (candidate.text, query.text))

    missing = [(key, pair) for key, pair in pairs.items() if key not in score_cache]
    reused_count = len(pairs) - len(missing)
    for offset in range(0, len(missing), _NLI_PAIRS_PER_BATCH):
        batch = missing[offset : offset + _NLI_PAIRS_PER_BATCH]
        directional_pairs = [
            direction
            for _, (candidate, query) in batch
            for direction in ((candidate, query), (query, candidate))
        ]
        values = await scorer.entailment_probabilities(directional_pairs)
        if len(values) != len(directional_pairs):
            raise RuntimeError(
                f"NLI scorer returned {len(values)} scores; "
                f"expected {len(directional_pairs)}"
            )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise RuntimeError("NLI scorer returned an invalid probability")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as output:
            for index, (key, _) in enumerate(batch):
                pair_scores = (values[index * 2], values[index * 2 + 1])
                score_cache[key] = pair_scores
                output.write(
                    json.dumps(
                        {
                            "key": key,
                            "candidate_to_query": pair_scores[0],
                            "query_to_candidate": pair_scores[1],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return reused_count, len(missing)


def _score_rows(
    query_examples: Sequence[IntentExample],
    candidate_examples: Sequence[IntentExample],
    candidate_rows: CandidateRows,
    nli_scores: Mapping[str, tuple[float, float]],
) -> list[list[dict[str, object]]]:
    output: list[list[dict[str, object]]] = []
    for query_index, candidates in enumerate(candidate_rows):
        query = query_examples[query_index]
        row: list[dict[str, object]] = []
        for candidate_index, cosine in candidates:
            candidate = candidate_examples[candidate_index]
            row.append(
                {
                    "candidate_index": candidate_index,
                    "cosine": cosine,
                    "stage0_pass": stage0_guard(candidate.text, query.text),
                    "nli_scores": nli_scores.get(_pair_key(candidate.text, query.text)),
                    "candidate_intent": candidate.intent,
                }
            )
        output.append(row)
    return output


def _evaluate(
    query_examples: Sequence[IntentExample],
    rows: Sequence[Sequence[Mapping[str, object]]],
    *,
    cosine_threshold: float,
    nli_threshold: float | None = None,
) -> dict[str, int | float | None]:
    hits = 0
    false_positives = 0
    for query, candidates in zip(query_examples, rows, strict=True):
        for candidate in candidates:
            if cast(float, candidate["cosine"]) < cosine_threshold:
                continue
            if not cast(bool, candidate["stage0_pass"]):
                continue
            if nli_threshold is not None:
                probabilities = candidate["nli_scores"]
                if probabilities is None:
                    continue
                forward, reverse = cast(tuple[float, float], probabilities)
                if forward < nli_threshold or reverse < nli_threshold:
                    continue
            hits += 1
            if candidate["candidate_intent"] != query.intent:
                false_positives += 1
            break
    same_intent_hits = hits - false_positives
    return {
        "hits": hits,
        "false_positives": false_positives,
        "same_intent_hits": same_intent_hits,
        "precision": same_intent_hits / hits if hits else None,
        "recall": same_intent_hits / len(query_examples) if query_examples else None,
    }


def _calibrate(
    train: Sequence[IntentExample],
    train_rows: Sequence[Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    operating_points: list[dict[str, object]] = []
    for cosine_threshold in _COSINE_THRESHOLDS:
        metrics = _evaluate(
            train,
            train_rows,
            cosine_threshold=cosine_threshold,
        )
        operating_points.append(
            {
                "configuration": "cosine_stage0",
                "cosine_threshold": cosine_threshold,
                "nli_threshold": None,
                **metrics,
            }
        )
    for cosine_threshold in _NLI_COSINE_THRESHOLDS:
        for nli_threshold in _NLI_THRESHOLDS:
            metrics = _evaluate(
                train,
                train_rows,
                cosine_threshold=cosine_threshold,
                nli_threshold=nli_threshold,
            )
            operating_points.append(
                {
                    "configuration": "stage0_nli",
                    "cosine_threshold": cosine_threshold,
                    "nli_threshold": nli_threshold,
                    **metrics,
                }
            )

    selected_rows: list[dict[str, object]] = []
    for target in _PRECISION_TARGETS:
        for configuration in ("cosine_stage0", "stage0_nli"):
            eligible = [
                row
                for row in operating_points
                if row["configuration"] == configuration
                and row["precision"] is not None
                and cast(float, row["precision"]) >= target
            ]
            selected = (
                max(
                    eligible,
                    key=lambda row: (
                        cast(float | None, row["recall"]) or 0.0,
                        cast(float, row["cosine_threshold"]),
                        cast(float, row["nli_threshold"])
                        if row["nli_threshold"] is not None
                        else 0.0,
                    ),
                )
                if eligible
                else None
            )
            selected_rows.append(
                {
                    "precision_target": target,
                    "configuration": configuration,
                    "calibration_status": (
                        "selected" if selected is not None else "unreachable"
                    ),
                    **(
                        {
                            "cosine_threshold": selected["cosine_threshold"],
                            "nli_threshold": selected["nli_threshold"],
                            "train_hits": selected["hits"],
                            "train_false_positives": selected["false_positives"],
                            "train_precision": selected["precision"],
                            "train_recall": selected["recall"],
                        }
                        if selected is not None
                        else {}
                    ),
                }
            )
    return selected_rows


def _test_selected_metrics(
    selected_rows: Sequence[Mapping[str, object]],
    test: Sequence[IntentExample],
    test_rows: Sequence[Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for selected in selected_rows:
        result = dict(selected)
        if selected["calibration_status"] == "selected":
            metrics = _evaluate(
                test,
                test_rows,
                cosine_threshold=cast(float, selected["cosine_threshold"]),
                nli_threshold=cast(float | None, selected.get("nli_threshold")),
            )
            result.update(
                {f"test_{key}": value for key, value in metrics.items()}
            )
        results.append(result)
    return results


def _cosine_sweep(
    test: Sequence[IntentExample],
    test_rows: Sequence[Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in _COSINE_THRESHOLDS:
        rows.append(
            {
                "cosine_threshold": threshold,
                "configuration": "cosine_only",
                **_evaluate_cosine_only(test, test_rows, threshold),
            }
        )
        rows.append(
            {
                "cosine_threshold": threshold,
                "configuration": "cosine_stage0",
                **_evaluate(test, test_rows, cosine_threshold=threshold),
            }
        )
    return rows


def _evaluate_cosine_only(
    queries: Sequence[IntentExample],
    rows: Sequence[Sequence[Mapping[str, object]]],
    threshold: float,
) -> dict[str, int | float | None]:
    hits = 0
    false_positives = 0
    for query, candidates in zip(queries, rows, strict=True):
        candidate = next(
            (
                candidate
                for candidate in candidates
                if cast(float, candidate["cosine"]) >= threshold
            ),
            None,
        )
        if candidate is None:
            continue
        hits += 1
        if candidate["candidate_intent"] != query.intent:
            false_positives += 1
    same_intent_hits = hits - false_positives
    return {
        "hits": hits,
        "false_positives": false_positives,
        "same_intent_hits": same_intent_hits,
        "precision": same_intent_hits / hits if hits else None,
        "recall": same_intent_hits / len(queries) if queries else None,
    }


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_report(
    *,
    train: Sequence[IntentExample],
    test: Sequence[IntentExample],
    test_selected: Sequence[Mapping[str, object]],
    cosine_rows: Sequence[Mapping[str, object]],
    fp_samples: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> dict[str, str]:
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    prefix = "banking77_loo_train_calibrated_test"
    selected_path = _RESULTS_DIRECTORY / f"{prefix}_matched_precision.csv"
    cosine_path = _RESULTS_DIRECTORY / f"{prefix}_cosine_sweep.csv"
    audit_path = _RESULTS_DIRECTORY / f"{prefix}_cosine_stage0_092_false_positives.csv"
    plot_path = _RESULTS_DIRECTORY / f"{prefix}_recall_vs_precision.png"
    report_path = _RESULTS_DIRECTORY / f"{prefix}.json"
    common = {
        "dataset": "PolyAI/banking77",
        "run_date_utc": metadata["run_date_utc"],
        "git_commit": metadata["git_commit"],
        "embedding_model": _EMBEDDING_MODEL,
        "fastembed_version": version("fastembed"),
        "nli_model": _NLI_MODEL,
        "nli_model_revision": _NLI_REVISION,
        "train_calibration_split": "train_leave_one_out",
        "test_evaluation_split": "test",
        "test_used_for_threshold_choice": False,
    }
    _write_csv(
        selected_path,
        [{**common, **row} for row in test_selected],
        tuple(common)
        + (
            "precision_target",
            "configuration",
            "calibration_status",
            "cosine_threshold",
            "nli_threshold",
            "train_hits",
            "train_false_positives",
            "train_precision",
            "train_recall",
            "test_hits",
            "test_false_positives",
            "test_same_intent_hits",
            "test_precision",
            "test_recall",
        ),
    )
    _write_csv(
        cosine_path,
        [{**common, **row} for row in cosine_rows],
        tuple(common)
        + (
            "cosine_threshold",
            "configuration",
            "hits",
            "false_positives",
            "same_intent_hits",
            "precision",
            "recall",
        ),
    )
    audit_rows: list[dict[str, object]] = []
    for rank, row in enumerate(fp_samples, start=1):
        candidate_index = cast(int, row["retrieved_candidate_index"])
        candidate_example = train[candidate_index]
        nli = cast(tuple[float, float] | None, row["nli_scores"])
        audit_rows.append(
            {
                **common,
                "sample_rank": rank,
                "query_text": row["query_text"],
                "candidate_text": candidate_example.text,
                "query_intent": row["query_intent"],
                "candidate_intent": row["candidate_intent"],
                "cosine_similarity": row["cosine_similarity"],
                "candidate_to_query_entailment": nli[0] if nli else None,
                "query_to_candidate_entailment": nli[1] if nli else None,
            }
        )
    _write_csv(
        audit_path,
        audit_rows,
        tuple(common)
        + (
            "sample_rank",
            "query_text",
            "candidate_text",
            "query_intent",
            "candidate_intent",
            "cosine_similarity",
            "candidate_to_query_entailment",
            "query_to_candidate_entailment",
        ),
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    for configuration, label, color in (
        ("cosine_only", "Cosine only", "tab:blue"),
        ("cosine_stage0", "Cosine + Stage-0", "tab:orange"),
    ):
        points = [
            row
            for row in cosine_rows
            if row["configuration"] == configuration and row["precision"] is not None
        ]
        points.sort(key=lambda row: cast(float, row["precision"]))
        axis.plot(
            [cast(float, row["precision"]) for row in points],
            [cast(float, row["recall"]) for row in points],
            marker="o",
            label=label,
            color=color,
        )
    target_markers = {0.95: "o", 0.97: "s", 0.98: "^"}
    for row in test_selected:
        if row.get("calibration_status") != "selected":
            continue
        axis.scatter(
            cast(float, row["test_precision"]),
            cast(float, row["test_recall"]),
            marker=target_markers[cast(float, row["precision_target"])],
            color=(
                "tab:green"
                if row["configuration"] == "stage0_nli"
                else "tab:orange"
            ),
            edgecolors="black",
            zorder=3,
        )
    axis.set(
        title="Banking77 test: train-LOO calibrated recall vs precision",
        xlabel="Test precision (same-intent hits / hits)",
        ylabel="Test recall (same-intent hits / queries)",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color="tab:blue",
                marker="o",
                label="Cosine only sweep",
            ),
            Line2D(
                [],
                [],
                color="tab:orange",
                marker="o",
                label="Cosine + Stage-0 sweep",
            ),
            Line2D(
                [],
                [],
                color="tab:green",
                marker="o",
                label="Stage-0 + NLI selected",
            ),
            Line2D(
                [],
                [],
                color="white",
                marker=target_markers[0.95],
                markeredgecolor="black",
                label="Target 0.95",
            ),
            Line2D(
                [],
                [],
                color="white",
                marker=target_markers[0.97],
                markeredgecolor="black",
                label="Target 0.97",
            ),
            Line2D(
                [],
                [],
                color="white",
                marker=target_markers[0.98],
                markeredgecolor="black",
                label="Target 0.98",
            ),
        ],
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    report_path.write_text(
        json.dumps(
            {
                **common,
                "train_count": len(train),
                "test_count": len(test),
                "embedding_model": _EMBEDDING_MODEL,
                "fastembed_version": version("fastembed"),
                "nli_model": _NLI_MODEL,
                "nli_model_revision": _NLI_REVISION,
                "precision_targets": list(_PRECISION_TARGETS),
                "cosine_sweep_thresholds": list(_COSINE_THRESHOLDS),
                "nli_cosine_thresholds": list(_NLI_COSINE_THRESHOLDS),
                "nli_thresholds": list(_NLI_THRESHOLDS),
                "threshold_selection": "train-only leave-one-out",
                "matched_precision_results": list(test_selected),
                "files": {
                    "matched_precision_csv": selected_path.name,
                    "cosine_sweep_csv": cosine_path.name,
                    "false_positive_audit_csv": audit_path.name,
                    "recall_vs_precision_png": plot_path.name,
                },
                **metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "matched_precision_csv": str(selected_path),
        "cosine_sweep_csv": str(cosine_path),
        "false_positive_audit_csv": str(audit_path),
        "recall_vs_precision_png": str(plot_path),
        "report_json": str(report_path),
    }


async def _run() -> dict[str, object]:
    train, test = _read_splits(None)
    run_date = datetime.now(UTC).isoformat()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    embedder = FastEmbedder()
    scorer = OnnxNliScorer()
    try:
        train_vectors, train_vectors_reused = await _cached_embeddings(
            train,
            split="train",
            embedder=embedder,
        )
        train_candidates = _top_candidates(
            train_vectors,
            train_vectors,
            leave_one_out=True,
        )
        nli_cache_path = _RESULTS_DIRECTORY / "banking77_nli_pair_scores.jsonl"
        nli_scores = _load_nli_cache(nli_cache_path)
        train_nli_reused, train_nli_computed = await _ensure_nli_scores(
            train,
            train,
            train_candidates,
            scorer=scorer,
            score_cache=nli_scores,
            cache_path=nli_cache_path,
        )
        train_rows = _score_rows(
            train,
            train,
            train_candidates,
            nli_scores=nli_scores,
        )
        selected = _calibrate(train, train_rows)

        test_vectors, test_vectors_reused = await _cached_embeddings(
            test,
            split="test",
            embedder=embedder,
        )
        test_candidates = _top_candidates(
            test_vectors,
            train_vectors,
            leave_one_out=False,
        )
        test_nli_reused, test_nli_computed = await _ensure_nli_scores(
            test,
            train,
            test_candidates,
            scorer=scorer,
            score_cache=nli_scores,
            cache_path=nli_cache_path,
        )
        test_rows: list[list[dict[str, object]]] = []
        for query_index, candidates in enumerate(test_candidates):
            row: list[dict[str, object]] = []
            for candidate_index, cosine in candidates:
                candidate = train[candidate_index]
                query = test[query_index]
                row.append(
                    {
                        "candidate_index": candidate_index,
                        "cosine": cosine,
                        "stage0_pass": stage0_guard(candidate.text, query.text),
                        "nli_scores": nli_scores.get(
                            _pair_key(candidate.text, query.text)
                        ),
                        "candidate_intent": candidate.intent,
                    }
                )
            test_rows.append(row)

        test_selected = _test_selected_metrics(selected, test, test_rows)
        cosine_rows = _cosine_sweep(test, test_rows)
        all_092_false_positives: list[dict[str, object]] = []
        for query, candidate_rows_for_query in zip(test, test_rows, strict=True):
            retrieved = next(
                (
                    row
                    for row in candidate_rows_for_query
                    if cast(float, row["cosine"]) >= 0.92
                    and cast(bool, row["stage0_pass"])
                ),
                None,
            )
            if retrieved is not None and retrieved["candidate_intent"] != query.intent:
                all_092_false_positives.append(
                    {
                        "query_text": query.text,
                        "query_intent": query.intent,
                        "cosine_similarity": retrieved["cosine"],
                        "retrieved_candidate_index": retrieved["candidate_index"],
                        "candidate_intent": retrieved["candidate_intent"],
                        "nli_scores": retrieved["nli_scores"],
                    }
                )
        sampled_092_false_positives = random.Random(_RANDOM_SEED).sample(
            all_092_false_positives,
            min(_FP_AUDIT_SIZE, len(all_092_false_positives)),
        )
        metadata: dict[str, object] = {
            "run_date_utc": run_date,
            "git_commit": git_commit,
            "embedding_cache": {
                "train_reused": train_vectors_reused,
                "test_reused": test_vectors_reused,
                "train_rows": len(train),
                "test_rows": len(test),
            },
            "nli_cache": {
                "train_reused": train_nli_reused,
                "train_computed": train_nli_computed,
                "test_reused": test_nli_reused,
                "test_computed": test_nli_computed,
                "cache_path": nli_cache_path.name,
            },
            "cosine_cache": "persistent split-specific compressed NumPy files",
            "test_split_used_for_threshold_choice": False,
        }
        files = _save_report(
            train=train,
            test=test,
            test_selected=test_selected,
            cosine_rows=cosine_rows,
            fp_samples=sampled_092_false_positives,
            metadata=metadata,
        )
        return {
            **metadata,
            "train_count": len(train),
            "test_count": len(test),
            "calibration_selection": selected,
            "test_matched_precision_results": test_selected,
            "test_cosine_sweep": cosine_rows,
            "cosine_stage0_092_false_positive_count": len(
                all_092_false_positives
            ),
            "cosine_stage0_092_false_positive_sample_count": len(
                sampled_092_false_positives
            ),
            "files": files,
        }
    finally:
        await scorer.close()
        await embedder.close()


def main() -> None:
    """Run Banking77 train-LOO calibration and test-only reporting."""
    _arguments()
    print(json.dumps(asyncio.run(_run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
