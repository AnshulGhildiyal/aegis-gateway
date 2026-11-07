# Evaluation results and follow-up

This file is the append-only log for measured Aegis evaluation outputs. Add a
new dated section when a notebook or benchmark output changes; do not replace
older results. Record the dataset/split, labeling rule, model/configuration,
run scope, measured values, derived values, issues, and follow-up actions.
Never fill gaps with estimated benchmark results.

## 2026-11-07 — STS-B cosine-threshold sweep

### Run details

- Dataset: Hugging Face `nyu-mll/glue`, configuration `stsb`, `validation`.
- Examples: 1,500 sentence pairs.
- Positive label: STS-B human similarity score `>= 4`; 264 pairs were positive.
- Sweep: 101 cosine thresholds from -1.00 through 1.00 in increments of 0.02.
- Embeddings: the notebook's configured `FastEmbedder` default,
  `BAAI/bge-small-en-v1.5`.
- Scope: cosine-only sentence-pair similarity. This run did **not** evaluate the
  Stage-0 guard, NLI reranker, Qdrant retrieval, or strict cache-key filters.
- Source: locally saved output from
  [the STS-B threshold-sweep notebook](../notebooks/sts_b_threshold_sweep.ipynb).
- Timing shown by that notebook run: embedding batch 48.724101 seconds; cosine
  scoring mean 0.007605 milliseconds per pair. These are environment-specific
  measurements, not general performance guarantees. Dataset download and model
  initialization are not included in the reported cosine mean.

### Measured and derived results

Precision, recall, false-positive rate, accuracy, and confusion counts below
were emitted by the notebook. F1 is derived from those counts using
`2 * TP / (2 * TP + FP + FN)`.

| Cosine threshold | Precision | Recall | F1 (derived) | False-positive rate | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.2330 | 1.0000 | 0.3780 | 0.7031 | 0.4207 |
| 0.70 | 0.2892 | 0.9924 | 0.4479 | 0.5210 | 0.5693 |
| 0.80 | 0.4370 | 0.9583 | 0.6002 | 0.2638 | 0.7753 |
| 0.84 | 0.5490 | 0.9129 | 0.6856 | 0.1602 | 0.8527 |
| 0.86 | 0.6128 | 0.8333 | 0.7063 | 0.1125 | 0.8780 |
| **0.88** | **0.6623** | **0.7576** | **0.7067** | **0.0825** | **0.8893** |
| 0.90 | 0.7143 | 0.6250 | 0.6667 | 0.0534 | 0.8900 |
| 0.92 | 0.8293 | 0.5152 | 0.6355 | 0.0227 | 0.8960 |
| 0.94 | 0.8898 | 0.3977 | 0.5497 | 0.0105 | 0.8853 |
| 0.96 | 0.9722 | 0.2652 | 0.4167 | 0.0016 | 0.8693 |
| 1.00 | 1.0000 | 0.0038 | 0.0075 | 0.0000 | 0.8247 |

The maximum F1 in this sweep was **0.7067 at threshold 0.88**, with precision
**0.6623** and recall **0.7576**. Threshold 0.86 was nearly tied at F1 0.7063,
with higher recall (0.8333) but lower precision (0.6128). Threshold 0.90 had
higher precision (0.7143) and lower false-positive rate (0.0534), but lower
recall (0.6250).

### Insights

- Raising the threshold reduced false positives and recall, showing a clear
  precision/recall trade-off for cosine-only matching.
- A threshold selected by maximum F1 is not automatically the right cache
  threshold: the relative cost of a false cache hit versus an upstream miss
  should guide the operating point.
- At threshold 0.88, 102 false positives and 64 false negatives were recorded
  (TP=200). This is not yet a safe semantic-cache acceptance result because the
  stricter Stage-0 and bidirectional NLI checks were not included.

### Issues and actions

1. **Validation-set threshold selection can overstate generalization.**
   - Action: use STS-B training data (or a dedicated calibration split) to pick
     a threshold, then report the selected threshold once on validation or a
     held-out test split. Keep the split and selection procedure fixed.
2. **STS-B similarity is not the same target as safe answer reuse.**
   - Action: treat this as an embedding baseline only. Evaluate false-positive
     cache hits on the requested PAWS/QQP adversarial pairs, Banking77 intent
     labels, and manually labeled LMSYS prompts. Apply Stage-0 and bidirectional
     NLI before considering a cache hit.
3. **F1 optimizes a symmetric classification objective, not gateway risk.**
   - Action: report false-positive rate and false-positive counts alongside
     precision/recall. Select the production operating threshold from an
     explicitly stated false-positive tolerance and validation evidence; do not
     adopt 0.88 solely because it maximized this sweep's F1.
4. **Single-run latency is machine- and state-dependent.**
   - Action: repeat timings after model warmup, document runtime/hardware and
     whether model download/initialization is included, and report a latency
     distribution. Keep cosine-only, embedding, and cosine + Stage-0 + NLI
     costs separate.

## 2026-11-07 — STS-B validation fine sweep

### Run details

- Dataset: Hugging Face `nyu-mll/glue`, configuration `stsb`, `validation` only.
- Examples: 1,500 pairs; 264 positives using the same `score >= 4` rule.
- Sweep: 20 thresholds, 0.80 through 0.99 in 0.01 increments.
- Embeddings: `BAAI/bge-small-en-v1.5`, served from the FastEmbed registry's
  `Qdrant/bge-small-en-v1.5-onnx-Q` source (`model_optimized.onnx`, 384
  dimensions); FastEmbed package version `0.9.1`.
- Run date: 2026-11-07 UTC. Git HEAD recorded as
  `5f63dfc3b360b270f1b86e06101b2bd4c9b51553`; the worktree was dirty, so the
  commit does not contain the uncommitted notebook/runner changes that produced
  this output. The registry source is recorded, but an immutable model artifact
  revision was not available in this run's metadata.
- Scope: cosine-only baseline. The test split, Stage-0 guard, NLI reranker, and
  cache retrieval were not evaluated.
- Re-run with all notebook cells or
  `uv run --group benchmark python -m benchmarks.stsb_fine_sweep`.
- Measured timing: embedding batch 47.024462 seconds; mean cosine calculation
  0.006497 milliseconds per pair. The latter excludes embedding time; neither
  value includes NLI because this run did not execute NLI.
- Artifacts:
  [fine-sweep CSV](../benchmarks/results/sts_b_validation_fine_sweep.csv),
  [error-analysis CSV](../benchmarks/results/sts_b_validation_error_analysis.csv),
  [threshold plot](../benchmarks/results/sts_b_validation_metrics_vs_threshold.png),
  [precision-recall plot](../benchmarks/results/sts_b_validation_precision_recall.png),
  and [run metadata](../benchmarks/results/sts_b_validation_metadata.json).

### Measured operating points

| Selection | Threshold | Precision | Recall | F1 | Predicted hits | False positives |
|---|---:|---:|---:|---:|---:|---:|
| Best F1 in fine sweep | **0.85** | 0.5899 | 0.8826 | **0.7071** | 395 | 162 |
| Lowest threshold with precision >= 0.90 | **0.95** | 0.9479 | 0.3447 | 0.5056 | 96 | 5 |
| Lowest threshold with precision >= 0.95 | **0.96** | 0.9722 | 0.2652 | 0.4167 | 72 | 2 |

The 0.85 best-F1 point is a narrow improvement over the earlier coarse sweep's
0.88 result (F1 0.7067). At 0.88 in this run, precision was 0.6623, recall
0.7576, with 302 predicted hits, 102 false positives, and 64 false negatives.

### Error analysis

- At 0.92, the requested 15 false positives were available and saved.
- At 0.96, all 2 false positives were printed and saved.
- At 0.88, 10 false negatives were printed and saved.
- The false-positive examples include sentence pairs with highly overlapping
  wording but meaningful added/removed details, as well as a role reversal
  example. The false negatives include paraphrases close to the threshold.
  This supports treating cosine as candidate retrieval, not as the final
  acceptance decision.

### Insights, issues, and actions

1. **High precision costs recall on this labeled similarity task.** Reaching
   precision >= 0.95 requires threshold 0.96 and leaves recall at 0.2652. Do not
   choose a threshold from F1 alone for cache reuse.
2. **Cosine-only false positives can change details or roles.** Continue to
   enforce the Stage-0 number/negation guard and bidirectional NLI before
   accepting semantic cache hits; validate those stages separately.
3. **The embedding artifact is not fully revision-pinned.** The model name,
   registry source, ONNX filename, FastEmbed version, date, and Git HEAD are
   recorded, but no immutable artifact revision/hash was captured. Pin or record
   an artifact checksum in a future reproducibility pass.
4. **The worktree was dirty when this run was captured.** Commit the notebook,
   runner, dependencies, and output metadata together (or rerun from a clean
   commit) for a fully source-reproducible result.

## Future result entries

For each new output, append another dated section with the exact command or
notebook, dataset revision/split, sample count, label definition, model versions,
thresholds, measured results, derived calculations, and any issues/actions.
Preserve previous entries so changes between runs can be compared honestly.
