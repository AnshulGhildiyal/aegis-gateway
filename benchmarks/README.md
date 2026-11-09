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

The same run includes a validation cosine-only and cosine + Stage-0 sweep from
0.85 to 0.99 in 0.01 increments. It also selects matched-precision operating
points for targets 0.80, 0.90, and 0.95 on the fixed-seed train sample, then
reports their validation hits, precision, and recall. NLI operating points are
selected separately at cosine 0.85 and 0.95. Unmet calibration targets are
marked `unreachable`. A recall-vs-precision plot and CSVs record these results.
Stage-0 diagnostics include counts over the full validation split and by each
pre-guard cosine candidate threshold, so the population for each count is
explicit.

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

The candidate pool is the train split and queries are from the existing test
split. The evaluator runs four frozen operating points taken from the QQP
train-calibrated results: cosine + Stage-0 at 0.98, Stage-0 + NLI at cosine
0.95 / NLI 0.97, and reference configs Stage-0 + NLI at 0.85 / 0.23 and cosine
+ Stage-0 at 0.92. No threshold is selected or adjusted on Banking77.
False-positive hits are accepted cache hits whose returned training example has
a different Banking77 intent label. Precision is same-intent hits divided by
all cache hits; recall is same-intent hits divided by test queries. NLI call
rate is the fraction of test queries with at least one Stage-0-passing
candidate sent to the NLI model. The run writes a per-config metrics CSV and a
top-15 false-positive intent-confusion CSV for each config.

```bash
uv run --group benchmark python -m benchmarks.evaluate_banking77
```

Use `--limit N` for a smoke run; omit it for the full splits. Outputs are saved
to `benchmarks/results/banking77_frozen_config_metrics.csv` and
`benchmarks/results/banking77_frozen_config_top_confusions.csv`.

### Banking77 train-LOO calibration

This evaluator selects thresholds using only training queries, with each query
compared against the remaining training examples (self-match excluded). It
reports the test cosine sweep for cosine-only and cosine + Stage-0, then
compares train-selected operating points at precision targets 0.95, 0.97, and
0.98 for cosine-only, cosine + Stage-0, and Stage-0 + bidirectional NLI. NLI
thresholds are selected on train leave-one-out data at cosine thresholds 0.85,
0.90, and 0.95.
Test examples are used only for final evaluation and the 40-row false-positive
audit sample at cosine + Stage-0 0.92. Embeddings and directional NLI scores
are cached across runs.

```bash
uv run --group benchmark python -m benchmarks.evaluate_banking77_loo
```

The runner writes matched-precision and cosine-sweep CSVs, a sampled
false-positive audit CSV, a recall-vs-precision PNG, and JSON run metadata under
`benchmarks/results/`. Its threshold selection and test results are logged in
[docs/results.md](../docs/results.md) after execution.

When all required split embeddings and directional NLI scores already exist,
use `--cached-scores-only`. This mode creates no model instances and fails
explicitly rather than computing a missing embedding or NLI score:

```bash
uv run --group benchmark python -m benchmarks.evaluate_banking77_loo \
  --cached-scores-only
```

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
