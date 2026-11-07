# Semantic-cache evaluation

Install the project and benchmark tools:

```bash
uv sync --dev --group benchmark
```

The evaluation scripts fetch datasets from Hugging Face at run time. They do not
contain benchmark results or synthetic substitutes. Outputs are measured from the
dataset split and thresholds supplied to each run.
Recorded runs, insights, limitations, and follow-up actions are kept in
[docs/results.md](../docs/results.md); append a dated entry whenever evaluation
outputs change.

## STS-B threshold-sweep notebook

Open [sts_b_threshold_sweep.ipynb](../notebooks/sts_b_threshold_sweep.ipynb) and
run its cells. It loads only the GLUE STS-B validation split and labels a pair
as similar when its human score is at least 4.0. It sweeps cosine thresholds
from 0.80 through 0.99 by 0.01, reports best F1 and the lowest thresholds
meeting precision >= 0.90 and >= 0.95, and prints the requested false-positive
and false-negative examples. It saves the threshold metrics and error examples
as CSV, precision/recall plots as PNG, and run metadata under
`benchmarks/results/`.

The same analysis can be run from a terminal:

```bash
uv run --group benchmark python -m benchmarks.stsb_fine_sweep
```

## PAWS and QQP ablation

The ablation runner calibrates on a fixed-seed reservoir sample of 5,000
training pairs per dataset and reports on PAWS `labeled_final` validation and
QQP GLUE validation. It sweeps NLI thresholds 0.01–0.99 at cosine thresholds
0.85 and 0.95, selects the maximum-recall threshold subject to precision
>= 0.90 and >= 0.95, and reports validation ablations and selected operating
points. It also writes NLI histograms, Stage-0 rejection counts by reason and
gold label, and 20 sampled Stage-0-rejected positive pairs per dataset. The
test split is never requested.

```bash
uv run --group benchmark python -m benchmarks.evaluate_paraphrase_ablation
```

By default this evaluates both datasets. Use `--dataset paws` or
`--dataset qqp` to run one dataset, or `--limit N` for a smaller smoke run. It
stores hash-keyed per-pair cosine/NLI score caches in `benchmarks/results/`,
reuses available cached scores, and computes NLI only for eligible cache
misses. Reports, CSVs, and histogram PNGs are also saved there.

The earlier single-operating-point evaluator is also validation-only:

```bash
uv run python -m benchmarks.evaluate_paraphrase \
  --dataset paws --cosine-threshold 0.85 --nli-threshold 0.9 --limit 100
uv run python -m benchmarks.evaluate_paraphrase \
  --dataset qqp --cosine-threshold 0.85 --nli-threshold 0.9 --limit 100
```

These threshold examples are command syntax only, not validated Aegis settings.
Do not use a test split until separately authorized.

## Banking77 intent evaluation

The candidate pool is the train split and queries are from the test split.
False-positive hits are accepted cache hits whose returned training example has
a different Banking77 intent label.

```bash
uv run python -m benchmarks.evaluate_banking77 \
  --cosine-threshold 0.8 --nli-threshold 0.9 --limit 100
```

Omit `--limit` for the full splits. Threshold values shown are examples only.
The output includes false-positive hits, their rate among accepted hits, a rate
relative to all test examples, stage outcomes, and measured embedding,
retrieval, and lookup latencies.

## LMSYS manual labeling sample

The LMSYS-Chat-1M repository requires agreeing to its Hugging Face access terms.
After accepting them, run:

```bash
uv run python -m benchmarks.sample_lmsys_for_labeling \
  --pair-count 100 --scan-limit 20000 \
  --output data/lmsys_manual_labels.csv
```

It streams a bounded number of rows, reservoir-samples distinct user prompts,
shuffles and pairs them, and leaves `manual_equivalence_label` and `notes`
blank. The resulting CSV contains user-submitted prompt text; keep it local and
handle it according to the dataset terms and your privacy requirements.
