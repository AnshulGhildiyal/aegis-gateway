"""Evaluate semantic cache false positives on PAWS and QQP."""

import argparse
import asyncio
import json
from collections.abc import Iterable, Mapping
from typing import cast

from datasets import load_dataset  # type: ignore[import-untyped]

from aegis.engine.embeddings import FastEmbedder
from aegis.engine.nli_reranker import NliReranker, OnnxNliScorer
from benchmarks.evaluation import evaluate_pair_examples, rows_to_pair_examples


def dataset_spec(
    name: str,
) -> tuple[str, str, str, str, str]:
    """Return repository, config, split, and paired text columns for a dataset."""
    if name == "paws":
        return (
            "google-research-datasets/paws",
            "labeled_final",
            "validation",
            "sentence1",
            "sentence2",
        )
    if name == "qqp":
        return ("nyu-mll/glue", "qqp", "validation", "question1", "question2")
    raise ValueError("dataset must be 'paws' or 'qqp'")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PAWS or QQP paraphrase pairs through the cache stages."
    )
    parser.add_argument("--dataset", choices=("paws", "qqp"), required=True)
    parser.add_argument("--cosine-threshold", type=float, required=True)
    parser.add_argument("--nli-threshold", type=float, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N examples; omit for the complete split.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    repository, config, split, text_a_field, text_b_field = dataset_spec(args.dataset)
    dataset = load_dataset(repository, config, split=split)
    examples = rows_to_pair_examples(
        cast(Iterable[Mapping[str, object]], dataset),
        text_a_field=text_a_field,
        text_b_field=text_b_field,
        label_field="label",
        limit=args.limit,
    )
    embedder = FastEmbedder()
    scorer = OnnxNliScorer()
    reranker = NliReranker(
        scorer,
        entailment_threshold=args.nli_threshold,
    )
    try:
        warmup_started = asyncio.get_running_loop().time()
        await scorer.entailment_probabilities(
            [("A short warmup premise.", "A short warmup hypothesis.")]
        )
        warmup_seconds = asyncio.get_running_loop().time() - warmup_started
        report = await evaluate_pair_examples(
            examples,
            embedder=embedder,
            reranker=reranker,
            cosine_threshold=args.cosine_threshold,
        )
    finally:
        await reranker.close()
        await embedder.close()
    return {
        "dataset": args.dataset,
        "split": split,
        "cosine_threshold": args.cosine_threshold,
        "nli_threshold": args.nli_threshold,
        "nli_model_warmup_seconds": warmup_seconds,
        **report.as_dict(),
    }


def main() -> None:
    """Load a labeled Hugging Face split and print measured metrics."""
    print(json.dumps(asyncio.run(_run(_arguments())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
