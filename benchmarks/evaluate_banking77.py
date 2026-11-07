"""Evaluate false-positive semantic hits across Banking77 intents."""

import argparse
import asyncio
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import numpy as np
from datasets import ClassLabel, load_dataset  # type: ignore[import-untyped]

from aegis.engine.embeddings import FastEmbedder
from aegis.engine.lookup import LookupMetric, SemanticLookup
from aegis.engine.nli_reranker import NliReranker, OnnxNliScorer
from aegis.storage.cache import CacheEntry, CacheKey
from benchmarks.evaluation import (
    IntentExample,
    duration_summary_ms,
    intent_hit_metrics,
    outcome_counts,
    rows_to_intent_examples,
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
        description="Measure strict cache hits against Banking77 intent labels."
    )
    parser.add_argument("--cosine-threshold", type=float, required=True)
    parser.add_argument("--nli-threshold", type=float, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit train and test splits for a quick run; omit for full splits.",
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


async def _run(args: argparse.Namespace) -> dict[str, object]:
    train, test = _read_splits(args.limit)
    embedder = FastEmbedder()
    nli_scorer = OnnxNliScorer()
    nli_reranker = NliReranker(
        nli_scorer,
        entailment_threshold=args.nli_threshold,
    )
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
        events: list[LookupMetric] = []
        lookup = SemanticLookup(
            cache,
            nli_reranker,
            cosine_threshold=args.cosine_threshold,
            metrics_hook=events.append,
        )
        key = CacheKey(
            system_prompt_hash="banking77-evaluation",
            tenant_id="banking77-evaluation",
            embedding_model_version=embedder.model_name,
            target_llm="banking77-evaluation",
            temperature=0.0,
            max_tokens=None,
        )
        hit_intents: list[str | None] = []
        retrieval_seconds: list[float] = []
        lookup_seconds: list[float] = []
        for example, vector in zip(test, test_vectors, strict=True):
            cache.query_vector = vector
            started = time.perf_counter()
            result = await lookup.lookup(example.text, key)
            lookup_seconds.append(time.perf_counter() - started)
            retrieval_seconds.append(cache.last_search_seconds)
            returned_intent = (
                result.entry.response.get("intent")
                if result.entry is not None
                else None
            )
            if returned_intent is not None and not isinstance(returned_intent, str):
                raise RuntimeError("cache response contains a non-string intent")
            hit_intents.append(returned_intent)

        hit_metrics = intent_hit_metrics(
            [example.intent for example in test],
            hit_intents,
        )
        return {
            "dataset": "PolyAI/banking77",
            "train_count": len(train),
            "test_count": len(test),
            "cosine_threshold": args.cosine_threshold,
            "nli_threshold": args.nli_threshold,
            "nli_model_warmup_seconds": nli_warmup_seconds,
            **hit_metrics,
            "stage_outcomes": outcome_counts(events),
            "latency_ms": {
                "training_embedding_batch": duration_summary_ms(
                    [train_embedding_seconds]
                ),
                "test_embedding_batch": duration_summary_ms([test_embedding_seconds]),
                "cosine_retrieval_per_query": duration_summary_ms(retrieval_seconds),
                "cosine_plus_stage0_nli_lookup_per_query": duration_summary_ms(
                    lookup_seconds
                ),
            },
        }
    finally:
        await nli_reranker.close()
        await embedder.close()


def main() -> None:
    """Run Banking77 retrieval evaluation and print measured metrics."""
    print(json.dumps(asyncio.run(_run(_arguments())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
