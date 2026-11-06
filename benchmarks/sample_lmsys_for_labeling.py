"""Sample LMSYS user-prompt pairs into a local manual-labeling CSV."""

import argparse
import csv
import json
import random
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from datasets import load_dataset  # type: ignore[import-untyped]

_DATASET = "lmsys/lmsys-chat-1m"


@dataclass(frozen=True, slots=True)
class PromptPair:
    """Unlabeled prompt pair for human equivalence review."""

    candidate_prompt: str
    query_prompt: str


def first_user_prompt(conversation: object) -> str | None:
    """Extract the first non-empty user turn from LMSYS conversation data."""
    parsed = conversation
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, list):
        raise ValueError("LMSYS conversation must be a list of message objects")
    for message in parsed:
        if not isinstance(message, dict):
            raise ValueError("LMSYS conversation messages must be objects")
        role = message.get("role")
        content = message.get("content")
        if (
            isinstance(role, str)
            and role in {"user", "human"}
            and isinstance(content, str)
            and content.strip()
        ):
            return content.strip()
    return None


def reservoir_sample_prompts(
    rows: Iterable[Mapping[str, object]],
    *,
    prompt_count: int,
    scan_limit: int,
    seed: int,
) -> tuple[list[str], int]:
    """Reservoir-sample distinct user prompts from at most ``scan_limit`` rows."""
    if prompt_count < 1:
        raise ValueError("prompt_count must be at least 1")
    if scan_limit < 1:
        raise ValueError("scan_limit must be at least 1")
    generator = random.Random(seed)
    reservoir: list[str] = []
    seen: set[str] = set()
    scanned = 0
    for row in rows:
        if scanned >= scan_limit:
            break
        scanned += 1
        prompt = first_user_prompt(row.get("conversation"))
        if prompt is None or prompt in seen:
            continue
        seen.add(prompt)
        if len(reservoir) < prompt_count:
            reservoir.append(prompt)
        else:
            replacement = generator.randrange(len(seen))
            if replacement < prompt_count:
                reservoir[replacement] = prompt
    return reservoir, scanned


def make_prompt_pairs(prompts: list[str], *, seed: int) -> list[PromptPair]:
    """Shuffle sampled prompts and pair them for independent human judgments."""
    shuffled = list(prompts)
    random.Random(seed).shuffle(shuffled)
    return [
        PromptPair(candidate_prompt=shuffled[index], query_prompt=shuffled[index + 1])
        for index in range(0, len(shuffled) - 1, 2)
    ]


def write_labeling_csv(pairs: Iterable[PromptPair], output_path: Path) -> int:
    """Write pairs and empty human-label fields to a local CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "pair_id",
                "candidate_prompt",
                "query_prompt",
                "manual_equivalence_label",
                "notes",
            ],
        )
        writer.writeheader()
        for count, pair in enumerate(pairs, start=1):
            writer.writerow(
                {
                    "pair_id": f"pair_{count:06d}",
                    "candidate_prompt": pair.candidate_prompt,
                    "query_prompt": pair.query_prompt,
                    "manual_equivalence_label": "",
                    "notes": "",
                }
            )
    return count


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample random LMSYS prompt pairs for local manual labeling."
    )
    parser.add_argument("--pair-count", type=int, default=100)
    parser.add_argument("--scan-limit", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lmsys_manual_labels.csv"),
    )
    return parser.parse_args()


def main() -> None:
    """Stream LMSYS, sample prompt pairs, and write an unlabeled local CSV."""
    args = _arguments()
    if args.pair_count < 1 or args.scan_limit < 1:
        raise ValueError("--pair-count and --scan-limit must be at least 1")
    print(
        "LMSYS-Chat-1M access is gated by Hugging Face dataset terms; "
        "accept them before downloading. The CSV contains user prompts; keep it "
        "local unless you have approval to share the data.",
        file=sys.stderr,
    )
    dataset = load_dataset(_DATASET, split="train", streaming=True)
    prompts, scanned = reservoir_sample_prompts(
        cast(Iterable[Mapping[str, object]], dataset),
        prompt_count=args.pair_count * 2,
        scan_limit=args.scan_limit,
        seed=args.seed,
    )
    pairs = make_prompt_pairs(prompts, seed=args.seed)
    written = write_labeling_csv(pairs, args.output)
    print(
        json.dumps(
            {
                "dataset": _DATASET,
                "rows_scanned": scanned,
                "distinct_prompts_sampled": len(prompts),
                "pairs_written": written,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
