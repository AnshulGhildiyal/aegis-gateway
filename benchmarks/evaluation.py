"""Reusable dataset evaluation and latency measurement helpers."""

import math
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from aegis.engine.lookup import (
    CandidateReranker,
    LookupMetric,
    SemanticLookup,
)
from aegis.engine.nli_reranker import NliReranker
from aegis.storage.cache import CacheEntry, CacheKey


class TextEmbedder(Protocol):
    """Embedding API used by evaluation helpers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text."""


class _SingleCandidateCache:
    """Supply one labeled candidate to the real lookup pipeline."""

    def __init__(self) -> None:
        self.entry: CacheEntry | None = None

    async def search(
        self,
        *,
        user_content: str,
        key: CacheKey,
        limit: int = 2,
    ) -> list[CacheEntry]:
        del user_content, key
        if self.entry is None or limit < 1:
            return []
        return [self.entry]


@dataclass(frozen=True, slots=True)
class PairExample:
    """A labeled semantic pair, with candidate stored before query."""

    candidate: str
    query: str
    label: bool


@dataclass(frozen=True, slots=True)
class PairScore:
    """Pair and its measured cosine similarity."""

    example: PairExample
    cosine_similarity: float


@dataclass(frozen=True, slots=True)
class PairEvaluation:
    """Measured metrics for cosine-only and full semantic lookup decisions."""

    example_count: int
    cosine_only: dict[str, int | float | None]
    cosine_plus_stage0_nli: dict[str, int | float | None]
    latency_ms: dict[str, dict[str, float | None]]
    stage_outcomes: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready result without inventing missing metrics."""
        return {
            "example_count": self.example_count,
            "cosine_only": self.cosine_only,
            "cosine_plus_stage0_nli": self.cosine_plus_stage0_nli,
            "latency_ms": self.latency_ms,
            "stage_outcomes": self.stage_outcomes,
        }


def binary_metrics(
    labels: Sequence[bool],
    predictions: Sequence[bool],
) -> dict[str, int | float | None]:
    """Compute binary classification metrics with explicit empty denominators."""
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")

    pairs = zip(labels, predictions, strict=True)
    true_positive = sum(label and prediction for label, prediction in pairs)
    pairs = zip(labels, predictions, strict=True)
    true_negative = sum(not label and not prediction for label, prediction in pairs)
    pairs = zip(labels, predictions, strict=True)
    false_positive = sum(not label and prediction for label, prediction in pairs)
    pairs = zip(labels, predictions, strict=True)
    false_negative = sum(label and not prediction for label, prediction in pairs)
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    actual_negative = true_negative + false_positive
    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": (
            true_positive / predicted_positive if predicted_positive else None
        ),
        "recall": true_positive / actual_positive if actual_positive else None,
        "false_positive_rate": (
            false_positive / actual_negative if actual_negative else None
        ),
        "accuracy": ((true_positive + true_negative) / len(labels) if labels else None),
    }


def stsb_binary_labels(scores: Sequence[float]) -> list[bool]:
    """Binarize STS-B human scores using the requested score >= 4 rule."""
    if any(not math.isfinite(score) or not 0 <= score <= 5 for score in scores):
        raise ValueError("STS-B similarity scores must be finite values from 0 to 5")
    return [score >= 4 for score in scores]


def threshold_sweep(
    labels: Sequence[bool],
    similarities: Sequence[float],
    thresholds: Iterable[float],
) -> list[dict[str, int | float | None]]:
    """Evaluate cosine thresholds against fixed binary pair labels."""
    if len(labels) != len(similarities):
        raise ValueError("labels and similarities must have equal length")
    if any(not math.isfinite(similarity) for similarity in similarities):
        raise ValueError("cosine similarities must all be finite")
    tolerance = 1e-6
    minimum = min(similarities, default=0.0)
    maximum = max(similarities, default=0.0)
    if minimum < -1 - tolerance or maximum > 1 + tolerance:
        raise ValueError(
            "cosine similarities must be in [-1, 1]; "
            f"observed range [{minimum:.9g}, {maximum:.9g}]"
        )
    bounded_similarities = [
        min(1.0, max(-1.0, similarity)) for similarity in similarities
    ]
    results: list[dict[str, int | float | None]] = []
    for threshold in thresholds:
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("cosine thresholds must be finite values from -1 to 1")
        metrics = binary_metrics(
            labels,
            [similarity >= threshold for similarity in bounded_similarities],
        )
        results.append({"threshold": threshold, **metrics})
    return results


async def score_pair_cosines(
    examples: Sequence[PairExample],
    embedder: TextEmbedder,
) -> tuple[list[PairScore], float, list[float]]:
    """Batch-embed pair texts and return cosine scores and measured timings.

    Returns the embedding batch duration and one cosine-computation duration
    per pair. Embedding time is reported separately from scoring latency.
    """
    if not examples:
        return [], 0.0, []
    texts = [
        text for example in examples for text in (example.candidate, example.query)
    ]
    embedding_started = time.perf_counter()
    vectors = await embedder.embed(texts)
    embedding_seconds = time.perf_counter() - embedding_started
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"embedder returned {len(vectors)} vectors; expected {len(texts)}"
        )
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError("embedder returned invalid vectors")
    norms = np.linalg.norm(matrix, axis=1)
    normalized = np.divide(
        matrix,
        norms[:, np.newaxis],
        out=np.zeros_like(matrix),
        where=norms[:, np.newaxis] != 0,
    )

    scores: list[PairScore] = []
    cosine_seconds: list[float] = []
    for index, example in enumerate(examples):
        started = time.perf_counter()
        similarity = float(
            np.clip(
                np.dot(normalized[index * 2], normalized[index * 2 + 1]),
                -1.0,
                1.0,
            )
        )
        cosine_seconds.append(time.perf_counter() - started)
        scores.append(PairScore(example, similarity))
    return scores, embedding_seconds, cosine_seconds


async def evaluate_pair_examples(
    examples: Sequence[PairExample],
    *,
    embedder: TextEmbedder,
    reranker: CandidateReranker | NliReranker,
    cosine_threshold: float,
    metrics_hook: list[LookupMetric] | None = None,
) -> PairEvaluation:
    """Evaluate exact pair candidates with cosine and the production lookup stages."""
    if not math.isfinite(cosine_threshold) or not -1 <= cosine_threshold <= 1:
        raise ValueError("cosine_threshold must be between -1 and 1")
    pair_scores, embedding_seconds, cosine_seconds = await score_pair_cosines(
        examples,
        embedder,
    )
    labels = [score.example.label for score in pair_scores]
    cosine_predictions = [
        score.cosine_similarity >= cosine_threshold for score in pair_scores
    ]

    cache = _SingleCandidateCache()
    events: list[LookupMetric] = []
    lookup = SemanticLookup(
        cache,
        reranker,
        cosine_threshold=cosine_threshold,
        metrics_hook=events.append,
    )
    strict_predictions: list[bool] = []
    full_lookup_seconds: list[float] = []
    key = CacheKey(
        system_prompt_hash="dataset-evaluation",
        tenant_id="dataset-evaluation",
        embedding_model_version="precomputed",
        target_llm="dataset-evaluation",
        temperature=0.0,
        max_tokens=None,
    )
    for score in pair_scores:
        cache.entry = CacheEntry(
            user_content=score.example.candidate,
            response={"label": score.example.label},
            created_at=0.0,
            expires_at=1.0,
            last_accessed_at=0.0,
            similarity=score.cosine_similarity,
        )
        started = time.perf_counter()
        result = await lookup.lookup(score.example.query, key)
        full_lookup_seconds.append(time.perf_counter() - started)
        strict_predictions.append(result.hit)

    if metrics_hook is not None:
        metrics_hook.extend(events)

    return PairEvaluation(
        example_count=len(pair_scores),
        cosine_only=binary_metrics(labels, cosine_predictions),
        cosine_plus_stage0_nli=binary_metrics(labels, strict_predictions),
        latency_ms={
            "embedding_batch": duration_summary_ms([embedding_seconds]),
            "cosine_only_per_pair": duration_summary_ms(cosine_seconds),
            "cosine_plus_stage0_nli_per_pair": duration_summary_ms(
                [
                    cosine_time + lookup_time
                    for cosine_time, lookup_time in zip(
                        cosine_seconds,
                        full_lookup_seconds,
                        strict=True,
                    )
                ]
            ),
        },
        stage_outcomes=outcome_counts(events),
    )


def duration_summary_ms(seconds: Sequence[float]) -> dict[str, float | None]:
    if not seconds:
        return {"mean": None, "p50": None, "p95": None}
    values = np.asarray(seconds, dtype=np.float64) * 1000
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def intent_hit_metrics(
    query_intents: Sequence[str],
    hit_intents: Sequence[str | None],
) -> dict[str, int | float | None]:
    """Count accepted hits whose candidate intent differs from the query intent."""
    if len(query_intents) != len(hit_intents):
        raise ValueError("query_intents and hit_intents must have equal length")
    hits = 0
    false_positive_hits = 0
    for query_intent, hit_intent in zip(query_intents, hit_intents, strict=True):
        if hit_intent is None:
            continue
        hits += 1
        if hit_intent != query_intent:
            false_positive_hits += 1
    return {
        "hits": hits,
        "same_intent_hits": hits - false_positive_hits,
        "false_positive_hits_different_intent": false_positive_hits,
        "false_positive_rate_of_hits": (false_positive_hits / hits if hits else None),
        "false_positive_rate_of_test_examples": (
            false_positive_hits / len(query_intents) if query_intents else None
        ),
    }


@dataclass(frozen=True, slots=True)
class IntentExample:
    """One utterance with its Banking77 intent label."""

    text: str
    intent: str


def rows_to_intent_examples(
    rows: Iterable[Mapping[str, object]],
    *,
    label_names: Sequence[str],
    text_field: str = "text",
    label_field: str = "label",
    limit: int | None = None,
) -> list[IntentExample]:
    """Validate Banking77 rows and resolve ClassLabel indices to intent names."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    examples: list[IntentExample] = []
    for row in rows:
        if limit is not None and len(examples) >= limit:
            break
        text = row.get(text_field)
        raw_label = row.get(label_field)
        if not isinstance(text, str):
            raise ValueError(f"{text_field} must be a string")
        if isinstance(raw_label, int) and not isinstance(raw_label, bool):
            if not 0 <= raw_label < len(label_names):
                raise ValueError("intent label index is outside label_names")
            intent = label_names[raw_label]
        elif isinstance(raw_label, str):
            intent = raw_label
        else:
            raise ValueError(f"{label_field} must be an intent string or index")
        examples.append(IntentExample(text=text, intent=intent))
    return examples


def rows_to_pair_examples(
    rows: Iterable[Mapping[str, object]],
    *,
    text_a_field: str,
    text_b_field: str,
    label_field: str,
    limit: int | None = None,
) -> list[PairExample]:
    """Validate labeled paraphrase rows from PAWS or GLUE/QQP."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    examples: list[PairExample] = []
    for row in rows:
        if limit is not None and len(examples) >= limit:
            break
        text_a = row.get(text_a_field)
        text_b = row.get(text_b_field)
        label = row.get(label_field)
        if not isinstance(text_a, str) or not isinstance(text_b, str):
            raise ValueError("pair text fields must be strings")
        if isinstance(label, bool):
            label_value = label
        elif isinstance(label, int) and label in (0, 1):
            label_value = label == 1
        else:
            raise ValueError("pair label field must contain binary 0/1 labels")
        examples.append(PairExample(candidate=text_a, query=text_b, label=label_value))
    return examples


def outcome_counts(events: Sequence[LookupMetric]) -> dict[str, dict[str, int]]:
    """Group stage outcomes into JSON-ready counts."""
    grouped: dict[str, Counter[str]] = {}
    for event in events:
        grouped.setdefault(event.stage, Counter())[event.outcome] += 1
    return {stage: dict(counts) for stage, counts in sorted(grouped.items())}
