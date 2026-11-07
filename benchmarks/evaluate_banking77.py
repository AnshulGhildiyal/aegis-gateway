"""Evaluate false-positive semantic hits across Banking77 intents."""

import argparse
import asyncio
import csv
import json
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
from datasets import ClassLabel, load_dataset  # type: ignore[import-untyped]

from aegis.engine.embeddings import FastEmbedder
from aegis.engine.nli_reranker import NliReranker, OnnxNliScorer
from aegis.engine.regex_guard import stage0_guard
from aegis.storage.cache import CacheEntry, CacheKey
from benchmarks.evaluation import (
    IntentExample,
    duration_summary_ms,
    intent_hit_metrics,
    rows_to_intent_examples,
)

_RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"
_TOP_CONFUSIONS = 15


@dataclass(frozen=True, slots=True)
class BankingEvaluationConfig:
    """One frozen cosine/Stage-0/NLI operating point."""

    name: str
    cosine_threshold: float
    nli_threshold: float | None


_FROZEN_CONFIGS = (
    BankingEvaluationConfig("cosine_stage0_c0.98", 0.98, None),
    BankingEvaluationConfig("cosine_stage0_nli_c0.95_n0.97", 0.95, 0.97),
    BankingEvaluationConfig("cosine_stage0_nli_c0.85_n0.23", 0.85, 0.23),
    BankingEvaluationConfig("cosine_stage0_c0.92", 0.92, None),
)


class BankingVectorCache:
    """In-memory candidate corpus over precomputed Banking77 embeddings."""

    def __init__(
        self,
        examples: Sequence[IntentExample],
        vectors: np.ndarray,
    ) -> None:
        if len(examples) != vectors.shape[0]:
            raise ValueError("one vector is required for every training example")
        self.examples = examples
        self.vectors = vectors
        self.query_vector = np.zeros(vectors.shape[1], dtype=np.float64)
        self.last_search_seconds = 0.0

    async def search(
        self,
        *,
        user_content: str,
        key: CacheKey,
        limit: int = 2,
    ) -> list[CacheEntry]:
        del user_content, key
        if limit < 1:
            raise ValueError("limit must be at least 1")
        started = time.perf_counter()
        similarities = np.clip(self.vectors @ self.query_vector, -1.0, 1.0)
        indices = np.argsort(similarities)[::-1][:limit]
        self.last_search_seconds = time.perf_counter() - started
        return [
            CacheEntry(
                user_content=self.examples[int(index)].text,
                response={"intent": self.examples[int(index)].intent},
                created_at=0.0,
                expires_at=1.0,
                last_accessed_at=0.0,
                similarity=float(similarities[index]),
            )
            for index in indices
        ]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen QQP-calibrated thresholds on Banking77 test queries."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit train and test splits for a quick smoke run; omit for full splits.",
    )
    return parser.parse_args()


def _read_splits(
    limit: int | None,
) -> tuple[list[IntentExample], list[IntentExample]]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    dataset = load_dataset("PolyAI/banking77")
    train_split = dataset["train"]
    test_split = dataset["test"]
    feature = train_split.features["label"]
    if not isinstance(feature, ClassLabel):
        raise ValueError("Banking77 label feature must be a ClassLabel")
    label_names = feature.names
    train = rows_to_intent_examples(
        cast(Iterable[Mapping[str, object]], train_split),
        label_names=label_names,
        limit=limit,
    )
    test = rows_to_intent_examples(
        cast(Iterable[Mapping[str, object]], test_split),
        label_names=label_names,
        limit=limit,
    )
    return train, test


def _normalize(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError("embedder returned invalid vectors")
    norms = np.linalg.norm(matrix, axis=1)
    return np.divide(
        matrix,
        norms[:, np.newaxis],
        out=np.zeros_like(matrix),
        where=norms[:, np.newaxis] != 0,
    )


async def _evaluate_config(
    config: BankingEvaluationConfig,
    *,
    queries: Sequence[IntentExample],
    query_vectors: np.ndarray,
    cache: BankingVectorCache,
    key: CacheKey,
    nli_scorer: OnnxNliScorer,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    reranker = (
        NliReranker(nli_scorer, entailment_threshold=config.nli_threshold)
        if config.nli_threshold is not None
        else None
    )
    returned_intents: list[str | None] = []
    nli_queries = 0
    nli_candidate_pairs = 0
    confusion_counts: Counter[tuple[str, str]] = Counter()

    for query, vector in zip(queries, query_vectors, strict=True):
        cache.query_vector = vector
        candidates = await cache.search(
            user_content=query.text,
            key=key,
            limit=2,
        )
        cosine_candidates = [
            candidate
            for candidate in candidates
            if candidate.similarity >= config.cosine_threshold
        ]
        stage0_candidates = [
            candidate
            for candidate in cosine_candidates
            if stage0_guard(candidate.user_content, query.text)
        ]

        accepted: CacheEntry | None = None
        if reranker is None:
            if stage0_candidates:
                accepted = stage0_candidates[0]
        elif stage0_candidates:
            nli_queries += 1
            nli_candidate_pairs += len(stage0_candidates)
            decisions = await reranker.rerank(query.text, stage0_candidates)
            accepted = next(
                (decision.candidate for decision in decisions if decision.accepted),
                None,
            )

        returned_intent = (
            accepted.response.get("intent") if accepted is not None else None
        )
        if returned_intent is not None and not isinstance(returned_intent, str):
            raise RuntimeError("cache response contains a non-string intent")
        returned_intents.append(returned_intent)
        if returned_intent is not None and returned_intent != query.intent:
            confusion_counts[(query.intent, returned_intent)] += 1

    hit_metrics = intent_hit_metrics(
        [query.intent for query in queries],
        returned_intents,
    )
    true_positive_hits = cast(int, hit_metrics["same_intent_hits"])
    hits = cast(int, hit_metrics["hits"])
    result: dict[str, object] = {
        "config": config.name,
        "cosine_threshold": config.cosine_threshold,
        "nli_threshold": config.nli_threshold,
        "query_count": len(queries),
        "hits": hits,
        "false_positives": hit_metrics["false_positive_hits_different_intent"],
        "true_intent_hits": true_positive_hits,
        "precision": true_positive_hits / hits if hits else None,
        "recall": true_positive_hits / len(queries) if queries else None,
        "nli_queries": nli_queries,
        "nli_call_rate": nli_queries / len(queries) if queries else None,
        "nli_candidate_pairs_scored": nli_candidate_pairs,
    }
    top_confusions = [
        {
            "config": config.name,
            "rank": rank,
            "query_intent": query_intent,
            "retrieved_intent": retrieved_intent,
            "false_positive_count": count,
        }
        for rank, ((query_intent, retrieved_intent), count) in enumerate(
            sorted(
                confusion_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )[:_TOP_CONFUSIONS],
            start=1,
        )
    ]
    return result, top_confusions


def _write_reports(
    summaries: Sequence[Mapping[str, object]],
    confusion_rows: Sequence[Mapping[str, object]],
    *,
    run_date: str,
    git_commit: str,
) -> tuple[Path, Path]:
    _RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary_path = _RESULTS_DIRECTORY / "banking77_frozen_config_metrics.csv"
    confusion_path = _RESULTS_DIRECTORY / "banking77_frozen_config_top_confusions.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "dataset",
                "run_date_utc",
                "git_commit",
                "candidate_split",
                "query_split",
                "config",
                "cosine_threshold",
                "nli_threshold",
                "query_count",
                "hits",
                "false_positives",
                "true_intent_hits",
                "precision",
                "recall",
                "nli_queries",
                "nli_call_rate",
                "nli_candidate_pairs_scored",
                "threshold_selection_source",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "dataset": "PolyAI/banking77",
                    "run_date_utc": run_date,
                    "git_commit": git_commit,
                    "candidate_split": "train",
                    "query_split": "test",
                    "threshold_selection_source": "frozen QQP train calibration",
                    **summary,
                }
            )
    with confusion_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "dataset",
                "run_date_utc",
                "git_commit",
                "query_split",
                "config",
                "rank",
                "query_intent",
                "retrieved_intent",
                "false_positive_count",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                "dataset": "PolyAI/banking77",
                "run_date_utc": run_date,
                "git_commit": git_commit,
                "query_split": "test",
                **row,
            }
            for row in confusion_rows
        )
    return summary_path, confusion_path


async def _run(args: argparse.Namespace) -> dict[str, object]:
    train, test = _read_splits(args.limit)
    run_date = datetime.now(UTC).isoformat()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    embedder = FastEmbedder()
    nli_scorer = OnnxNliScorer()
    try:
        started = time.perf_counter()
        train_vectors = _normalize(
            await embedder.embed([example.text for example in train])
        )
        train_embedding_seconds = time.perf_counter() - started
        started = time.perf_counter()
        test_vectors = _normalize(
            await embedder.embed([example.text for example in test])
        )
        test_embedding_seconds = time.perf_counter() - started

        nli_warmup_started = time.perf_counter()
        await nli_scorer.entailment_probabilities(
            [("A short warmup premise.", "A short warmup hypothesis.")]
        )
        nli_warmup_seconds = time.perf_counter() - nli_warmup_started
        cache = BankingVectorCache(train, train_vectors)
        key = CacheKey(
            system_prompt_hash="banking77-evaluation",
            tenant_id="banking77-evaluation",
            embedding_model_version=embedder.model_name,
            target_llm="banking77-evaluation",
            temperature=0.0,
            max_tokens=None,
        )
        summaries: list[dict[str, object]] = []
        confusion_rows: list[dict[str, object]] = []
        for config in _FROZEN_CONFIGS:
            summary, config_confusions = await _evaluate_config(
                config,
                queries=test,
                query_vectors=test_vectors,
                cache=cache,
                key=key,
                nli_scorer=nli_scorer,
            )
            summaries.append(summary)
            confusion_rows.extend(config_confusions)
        summary_path, confusion_path = _write_reports(
            summaries,
            confusion_rows,
            run_date=run_date,
            git_commit=git_commit,
        )
        return {
            "dataset": "PolyAI/banking77",
            "run_date_utc": run_date,
            "git_commit": git_commit,
            "train_count": len(train),
            "test_count": len(test),
            "thresholds_frozen_from": (
                "QQP 5,000-pair seeded train-calibrated evaluation"
            ),
            "nli_model_warmup_seconds": nli_warmup_seconds,
            "results": summaries,
            "top_confused_intent_pairs": confusion_rows,
            "summary_csv": str(summary_path),
            "confusion_pairs_csv": str(confusion_path),
            "latency_ms": {
                "training_embedding_batch": duration_summary_ms(
                    [train_embedding_seconds]
                ),
                "test_embedding_batch": duration_summary_ms([test_embedding_seconds]),
            },
            "test_split_used_for_threshold_choice": False,
        }
    finally:
        await nli_scorer.close()
        await embedder.close()


def main() -> None:
    """Run Banking77 retrieval evaluation and print measured metrics."""
    print(json.dumps(asyncio.run(_run(_arguments())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
