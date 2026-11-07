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

## 2026-11-07 — PAWS/QQP ablation (validation only)

### PAWS labeled_final

- Dataset: `google-research-datasets/paws`, configuration `labeled_final`,
  `validation` split only; 8,000 labeled pairs.
- Positive label: dataset label `1` means paraphrase.
- Cosine thresholds: 0.85, 0.88, 0.90, 0.92, and 0.95.
- NLI threshold: **0.89**, selected on this validation split by maximizing
  mean F1 across the five requested cosine thresholds; ties select the higher
  NLI threshold. This uses the same validation split for calibration and
  reported ablation metrics, so the results are not an independent estimate.
- Models: `BAAI/bge-small-en-v1.5` (FastEmbed 0.9.1) and
  `Xenova/nli-deberta-v3-small` at revision
  `6bc2a55c7c0f7e2bc68de60bb248e523e2612abb`.
- Run date: 2026-11-07 UTC. Git HEAD:
  `0d132f58abd2e26eeefa9d77ef032cd12271b833`; the worktree was dirty.
- The test split was not evaluated. Outputs:
  [PAWS JSON](../benchmarks/results/paraphrase_ablation_paws_validation.json),
  [metrics CSV](../benchmarks/results/paraphrase_ablation_paws_validation_metrics.csv),
  [false-positive samples CSV](../benchmarks/results/paraphrase_ablation_paws_validation_false_positives.csv).

| Cosine | Configuration | Hits | TP | FP | FN | Precision | Recall | F1 | NLI sent (count, % of split) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.85 | Cosine only | 7,960 | 3,518 | 4,442 | 21 | 0.4420 | 0.9941 | 0.6119 | — |
| 0.85 | Cosine + Stage-0 | 7,756 | 3,456 | 4,300 | 83 | 0.4456 | 0.9765 | 0.6120 | — |
| 0.85 | Cosine + Stage-0 + NLI | 5,396 | 3,103 | 2,293 | 436 | 0.5751 | 0.8768 | 0.6946 | 7,756; 96.95% |
| 0.88 | Cosine only | 7,905 | 3,496 | 4,409 | 43 | 0.4423 | 0.9878 | 0.6110 | — |
| 0.88 | Cosine + Stage-0 | 7,704 | 3,434 | 4,270 | 105 | 0.4457 | 0.9703 | 0.6109 | — |
| 0.88 | Cosine + Stage-0 + NLI | 5,372 | 3,089 | 2,283 | 450 | 0.5750 | 0.8728 | 0.6933 | 7,704; 96.30% |
| 0.90 | Cosine only | 7,834 | 3,467 | 4,367 | 72 | 0.4426 | 0.9797 | 0.6097 | — |
| 0.90 | Cosine + Stage-0 | 7,636 | 3,406 | 4,230 | 133 | 0.4460 | 0.9624 | 0.6096 | — |
| 0.90 | Cosine + Stage-0 + NLI | 5,342 | 3,073 | 2,269 | 466 | 0.5753 | 0.8683 | 0.6920 | 7,636; 95.45% |
| 0.92 | Cosine only | 7,704 | 3,427 | 4,277 | 112 | 0.4448 | 0.9684 | 0.6096 | — |
| 0.92 | Cosine + Stage-0 | 7,509 | 3,366 | 4,143 | 173 | 0.4483 | 0.9511 | 0.6093 | — |
| 0.92 | Cosine + Stage-0 + NLI | 5,267 | 3,041 | 2,226 | 498 | 0.5774 | 0.8593 | 0.6907 | 7,509; 93.86% |
| 0.95 | Cosine only | 7,107 | 3,236 | 3,871 | 303 | 0.4553 | 0.9144 | 0.6079 | — |
| 0.95 | Cosine + Stage-0 | 6,938 | 3,181 | 3,757 | 358 | 0.4585 | 0.8988 | 0.6072 | — |
| 0.95 | Cosine + Stage-0 + NLI | 4,970 | 2,908 | 2,062 | 631 | 0.5851 | 0.8217 | 0.6835 | 6,938; 86.73% |

False-positive sampling saved up to 15 examples for each cosine-threshold /
configuration combination (75 per configuration over the five thresholds).
Counts in the CSV are the measured values; displayed metric rates above are
rounded to four decimal places.

### QQP

- Dataset: `nyu-mll/glue`, configuration `qqp`, `validation` split only; 40,430
  labeled pairs. Positive label `1` means paraphrase.
- NLI threshold: **0.00**, selected by the same mean-F1 calibration rule over
  the five cosine thresholds on this validation split. At thresholds above
  0.00, mean F1 decreased; the next-best grid point was 0.01 with mean F1
  0.5646 versus 0.6060 at 0.00. At 0.00 every valid probability passes, so the
  NLI configuration is identical to cosine + Stage-0 for this selection
  criterion. This is a material issue, not evidence that NLI improves QQP.
- Run date: 2026-11-07 UTC. Git HEAD:
  `0d132f58abd2e26eeefa9d77ef032cd12271b833`; the worktree was dirty. The test
  split was not evaluated.
- Outputs:
  [QQP JSON](../benchmarks/results/paraphrase_ablation_qqp_validation.json),
  [metrics CSV](../benchmarks/results/paraphrase_ablation_qqp_validation_metrics.csv),
  [false-positive samples CSV](../benchmarks/results/paraphrase_ablation_qqp_validation_false_positives.csv).

| Cosine | Configuration | Hits | TP | FP | FN | Precision | Recall | F1 | NLI sent (count, % of split) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.85 | Cosine only | 17,775 | 11,938 | 5,837 | 2,947 | 0.6716 | 0.8020 | 0.7310 | — |
| 0.85 | Cosine + Stage-0 | 15,896 | 10,877 | 5,019 | 4,008 | 0.6843 | 0.7307 | 0.7067 | — |
| 0.85 | Cosine + Stage-0 + NLI | 15,896 | 10,877 | 5,019 | 4,008 | 0.6843 | 0.7307 | 0.7067 | 15,896; 39.32% |
| 0.88 | Cosine only | 13,976 | 10,095 | 3,881 | 4,790 | 0.7223 | 0.6782 | 0.6996 | — |
| 0.88 | Cosine + Stage-0 | 12,789 | 9,387 | 3,402 | 5,498 | 0.7340 | 0.6306 | 0.6784 | — |
| 0.88 | Cosine + Stage-0 + NLI | 12,789 | 9,387 | 3,402 | 5,498 | 0.7340 | 0.6306 | 0.6784 | 12,789; 31.63% |
| 0.90 | Cosine only | 11,275 | 8,590 | 2,685 | 6,295 | 0.7619 | 0.5771 | 0.6567 | — |
| 0.90 | Cosine + Stage-0 | 10,510 | 8,110 | 2,400 | 6,775 | 0.7716 | 0.5448 | 0.6387 | — |
| 0.90 | Cosine + Stage-0 + NLI | 10,510 | 8,110 | 2,400 | 6,775 | 0.7716 | 0.5448 | 0.6387 | 10,510; 26.00% |
| 0.92 | Cosine only | 8,707 | 6,978 | 1,729 | 7,907 | 0.8014 | 0.4688 | 0.5916 | — |
| 0.92 | Cosine + Stage-0 | 8,212 | 6,677 | 1,535 | 8,208 | 0.8131 | 0.4486 | 0.5782 | — |
| 0.92 | Cosine + Stage-0 + NLI | 8,212 | 6,677 | 1,535 | 8,208 | 0.8131 | 0.4486 | 0.5782 | 8,212; 20.31% |
| 0.95 | Cosine only | 5,021 | 4,334 | 687 | 10,551 | 0.8632 | 0.2912 | 0.4354 | — |
| 0.95 | Cosine + Stage-0 | 4,847 | 4,221 | 626 | 10,664 | 0.8708 | 0.2836 | 0.4278 | — |
| 0.95 | Cosine + Stage-0 + NLI | 4,847 | 4,221 | 626 | 10,664 | 0.8708 | 0.2836 | 0.4278 | 4,847; 11.99% |

False-positive sampling saved up to 15 examples for each cosine-threshold /
configuration combination (75 per configuration over the five thresholds).
Counts in the CSV are measured; displayed metric rates above are rounded to
four decimal places.

### Combined insights and next actions

1. On PAWS, NLI at the validation-selected 0.89 threshold improves precision
   from about 0.44–0.46 to 0.58–0.59, while reducing recall; false positives
   remain substantial (2,062–2,293 at these operating points).
2. On QQP, F1-based calibration selects 0.00 and NLI makes no change to the
   Stage-0-only predictions. Do **not** treat 0.00 as a production-safe
   entailment threshold: it accepts every scored candidate. The selected
   objective exposes that F1 alone is not adequate for choosing an NLI
   threshold for cache safety.
3. Use a stated risk constraint (for example a minimum precision or a maximum
   false-positive budget) for a future dev-only calibration pass. Keep the
   test split untouched until explicitly authorized. Because calibration and
   ablation were both run on validation, the reported metrics are not
   independent generalization estimates.
4. The PAWS and QQP runs both recorded a dirty worktree. Preserve the JSON
   metadata and commit the source changes before treating the run as fully
   source-reproducible.

## Future result entries

For each new output, append another dated section with the exact command or
notebook, dataset revision/split, sample count, label definition, model versions,
thresholds, measured results, derived calculations, and any issues/actions.
Preserve previous entries so changes between runs can be compared honestly.
