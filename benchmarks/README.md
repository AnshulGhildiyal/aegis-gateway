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
run its cells. It loads the GLUE STS-B validation split and labels a pair as
similar when its human score is at least 4.0. Cosine thresholds are swept on the
computed embeddings; the notebook displays the measured results in the session.

## PAWS and QQP

Supply both calibrated thresholds explicitly. PAWS uses the `labeled_final`
test split; QQP uses GLUE's validation split because its test labels are not
available for supervised evaluation.

```bash
uv run python -m benchmarks.evaluate_paraphrase \
  --dataset paws --cosine-threshold 0.8 --nli-threshold 0.9 --limit 100
uv run python -m benchmarks.evaluate_paraphrase \
  --dataset qqp --cosine-threshold 0.8 --nli-threshold 0.9 --limit 100
```

Omit `--limit` to evaluate the full labeled split. The threshold examples above
are command syntax only, not validated Aegis settings. Output reports the
cosine-only and full cosine + Stage-0 + NLI results separately, as well as
embedding-batch, cosine-only, and full lookup latencies. Latencies exclude the
dataset download; embedding time and first-use NLI model warmup are reported
separately from per-pair scoring latency.

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
