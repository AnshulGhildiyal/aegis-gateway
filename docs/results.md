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

## 2026-11-07 — PAWS/QQP train-calibrated NLI evaluation

### Protocol and artifacts

- Calibration used a fixed-seed reservoir sample of 5,000 `train` pairs per
  dataset (seed 42). Threshold sweeps used NLI thresholds 0.01–0.99 by 0.01
  and cosine thresholds 0.85 and 0.95. The threshold selection maximizes
  recall subject to the target precision; ties prefer the higher NLI threshold.
- Reported metrics and NLI score histograms use the validation split only.
  The test split was not requested or evaluated.
- Models: `BAAI/bge-small-en-v1.5` (FastEmbed 0.9.1) and
  `Xenova/nli-deberta-v3-small`, revision
  `6bc2a55c7c0f7e2bc68de60bb248e523e2612abb`.
- Git HEAD recorded for both runs:
  `937cb167f7ad3e848a9fd330bb4bfb262ccb9d49`; the worktree was dirty.
- PAWS validation has 8,000 pairs; QQP validation has 40,430 pairs. The
  validation NLI-score histograms include pairs with cosine >= 0.85 that passed
  Stage-0: 7,756 PAWS pairs and 15,896 QQP pairs. Each histogram shows both
  entailment directions and their minimum, grouped by gold label.
- Score caches were reused where exact pair/model keys were present. PAWS
  reused 3 train and 241 validation NLI scores; QQP loaded 211 prior sampled
  NLI scores, all reused on validation. New NLI inference was limited to
  missing eligible pairs.
- Output files:
  [PAWS report](../benchmarks/results/paraphrase_ablation_paws_train_calibrated_validation.json),
  [PAWS train sweep](../benchmarks/results/paraphrase_ablation_paws_train_calibrated_validation_train_nli_sweep.csv),
  [PAWS validation sweep](../benchmarks/results/paraphrase_ablation_paws_train_calibrated_validation_validation_nli_sweep.csv),
  [PAWS validation score histograms](../benchmarks/results/paraphrase_ablation_paws_nli_histograms.png),
  [QQP report](../benchmarks/results/paraphrase_ablation_qqp_train_calibrated_validation.json),
  [QQP train sweep](../benchmarks/results/paraphrase_ablation_qqp_train_calibrated_validation_train_nli_sweep.csv),
  [QQP validation sweep](../benchmarks/results/paraphrase_ablation_qqp_train_calibrated_validation_validation_nli_sweep.csv),
  and [QQP validation score histograms](../benchmarks/results/paraphrase_ablation_qqp_nli_histograms.png).
  The reports also link to selected-threshold metrics, Stage-0 breakdowns,
  rejected-positive samples, validation NLI scores, and the score caches.

### Calibration selections and validation results

| Dataset | Cosine | Precision target | Train-selected NLI threshold | Train precision | Train recall | Validation hits | Validation TP / FP / FN | Validation precision | Validation recall | Validation F1 | Validation pairs sent to NLI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PAWS | 0.85 | 0.90 | None | — | — | — | — | — | — | — | — |
| PAWS | 0.85 | 0.95 | None | — | — | — | — | — | — | — | — |
| PAWS | 0.95 | 0.90 | None | — | — | — | — | — | — | — | — |
| PAWS | 0.95 | 0.95 | None | — | — | — | — | — | — | — | — |
| QQP | 0.85 | 0.90 | 0.99 | 0.9043 | 0.0924 | 1,691 | 1,558 / 133 / 13,327 | 0.9213 | 0.1047 | 0.1880 | 15,896 (39.32%) |
| QQP | 0.85 | 0.95 | None | — | — | — | — | — | — | — | — |
| QQP | 0.95 | 0.90 | 0.97 | 0.9040 | 0.1485 | 2,526 | 2,355 / 171 / 12,530 | 0.9323 | 0.1582 | 0.2705 | 4,847 (11.99%) |
| QQP | 0.95 | 0.95 | None | — | — | — | — | — | — | — | — |

No PAWS threshold in the sweep reached precision >= 0.90. QQP reached
precision >= 0.90 at both cosine thresholds, but no threshold reached
precision >= 0.95. The QQP validation precision remained above 0.90 at the
train-selected thresholds, with low recall (0.1047 and 0.1582). Since the
thresholds were chosen on train and evaluated on validation, the reported
validation values are not used for further threshold tuning.

### Stage-0 rejection breakdown

The rejection counts in the original Stage-0 breakdown are by reason and gold
label over **all validation examples**, before applying any cosine threshold.
A pair may count under both reasons when both number and negation checks reject
it. The 20 saved rejected-positive examples were also sampled from that full
validation population.

| Dataset | Reason | Non-paraphrase rejections | Paraphrase rejections |
|---|---|---:|---:|
| PAWS | Number mismatch | 95 | 52 |
| PAWS | Negation mismatch | 49 | 13 |
| QQP | Number mismatch | 3,747 | 1,028 |
| QQP | Negation mismatch | 3,180 | 830 |

Twenty rejected true-paraphrase validation pairs per dataset were saved for
inspection. These counts show that the regex guard does reject some genuine
paraphrases, so its scope and false-rejection behavior need continued review.

### Insights, issues, and actions

1. **PAWS has no operating point meeting even 0.90 precision.** Do not promote
   an NLI threshold from this sweep for PAWS. Inspect the saved validation
   histograms and rejected-positive samples, and consider improving the model
   or the candidate/labeling protocol before another calibration run.
2. **QQP reaches 0.90 precision only with very low recall.** The 0.95 precision
   target is unmet at both cosine thresholds. Treat these as dataset-specific
   offline operating points, not safe production cache settings; validate on
   additional dev data and apply an explicit false-positive risk budget.
3. **Stage-0 rejects labeled positives.** Review the saved positive samples
   before expanding number/unit or negation rules. Any guard change should keep
   exact regression tests for the observed cases and be reevaluated on the
   same validation protocol.
4. **Reproducibility remains incomplete.** The worktree was dirty and dataset
   revisions were not pinned in the run metadata. Record immutable dataset
   revisions and rerun from a clean commit before treating these numbers as
   fully reproducible.

## 2026-11-07 — PAWS/QQP extended cosine sweep and matched precision

### Protocol and Stage-0 behavior

- Calibration used the existing deterministic 5,000-pair training sample
  (seed 42). All reported metrics and plots are from validation only. The test
  split was not requested or evaluated.
- The cosine-only and cosine + Stage-0 validation sweep uses thresholds 0.85
  through 0.99 in 0.01 increments. Matched-precision targets are 0.80, 0.90,
  and 0.95. Train calibration selects the highest-recall threshold meeting
  each target; ties prefer the higher cosine threshold, or higher NLI
  threshold. NLI options are calibrated separately at cosine 0.85 and 0.95
  over thresholds 0.01–0.99.
- **Stage-0 behavior did not change in the previous run.** Comparing the
  previous and current guard implementations shows the previous change only
  extracted number/negation rejection reasons: the allow condition remains
  equivalent to matching both extracted values. Diagnostics were extended,
  but the predicate was unchanged, so previously reported prediction rows did
  not need behavioral correction. The evaluation was rerun from cached scores.
- The original rejection breakdowns reported above cover the **entire
  validation split**, not only pairs retrieved by cosine. The new
  [candidate-set breakdowns](../benchmarks/results/) separately report all
  validation rows and the pre-Stage-0 candidate sets at every cosine threshold.
  At cosine 0.85, these sets contain 7,960 PAWS pairs and 17,775 QQP pairs.
  These are pairs with cosine >= threshold before the Stage-0 filter; reason
  counts can overlap.
- Models remain `BAAI/bge-small-en-v1.5` (FastEmbed 0.9.1) and
  `Xenova/nli-deberta-v3-small`, revision
  `6bc2a55c7c0f7e2bc68de60bb248e523e2612abb`. Both runs recorded Git HEAD
  `70270acea51eff6455e28e065a036789975b2880` with a dirty worktree.
- All cached scores were reused; neither embeddings nor NLI were recomputed.
  PAWS reused 5,000 train and 8,000 validation cosine scores, plus 4,853 train
  and 7,756 validation NLI pairs. QQP reused 5,000 train and 40,430 validation
  cosine scores, plus 1,953 train and 15,896 validation NLI pairs.
- Artifacts:
  [PAWS train-calibrated report](../benchmarks/results/paraphrase_ablation_paws_train_calibrated_validation.json),
  [PAWS cosine sweep](../benchmarks/results/paraphrase_ablation_paws_train_calibrated_validation_validation_cosine_sweep.csv),
  [PAWS matched-precision CSV](../benchmarks/results/paraphrase_ablation_paws_train_calibrated_validation_validation_matched_precision.csv),
  [PAWS recall-vs-precision plot](../benchmarks/results/paraphrase_ablation_paws_validation_matched_precision.png),
  [QQP train-calibrated report](../benchmarks/results/paraphrase_ablation_qqp_train_calibrated_validation.json),
  [QQP cosine sweep](../benchmarks/results/paraphrase_ablation_qqp_train_calibrated_validation_validation_cosine_sweep.csv),
  [QQP matched-precision CSV](../benchmarks/results/paraphrase_ablation_qqp_train_calibrated_validation_validation_matched_precision.csv),
  [QQP recall-vs-precision plot](../benchmarks/results/paraphrase_ablation_qqp_validation_matched_precision.png),
  and per-threshold
  [PAWS](../benchmarks/results/paraphrase_ablation_paws_train_calibrated_validation_stage0_candidate_set_rejections.csv)
  / [QQP](../benchmarks/results/paraphrase_ablation_qqp_train_calibrated_validation_stage0_candidate_set_rejections.csv)
  Stage-0 candidate-set counts.

### Cosine-only and cosine + Stage-0 sweep endpoints

The CSVs contain both configurations at all 15 cosine thresholds. The endpoints
below show the observed precision/recall trade-off; hits are measured on
validation.

| Dataset | Cosine threshold | Configuration | Hits | Precision | Recall |
|---|---:|---|---:|---:|---:|
| PAWS | 0.85 | Cosine only | 7,960 | 0.4420 | 0.9941 |
| PAWS | 0.85 | Cosine + Stage-0 | 7,756 | 0.4456 | 0.9765 |
| PAWS | 0.99 | Cosine only | 3,127 | 0.5510 | 0.4869 |
| PAWS | 0.99 | Cosine + Stage-0 | 3,083 | 0.5524 | 0.4812 |
| QQP | 0.85 | Cosine only | 17,775 | 0.6716 | 0.8020 |
| QQP | 0.85 | Cosine + Stage-0 | 15,896 | 0.6843 | 0.7307 |
| QQP | 0.99 | Cosine only | 980 | 0.9378 | 0.0617 |
| QQP | 0.99 | Cosine + Stage-0 | 972 | 0.9414 | 0.0615 |

### Matched-precision results

Thresholds were selected on the training sample and the selected operating
points below are measured on validation. Validation precision can differ from
the target because the target constrains calibration, not validation. “NLI”
rows are separate operating points for cosine 0.85 and 0.95.

**PAWS:** Every configuration/target combination was **unreachable on the
specified calibration grids**, including target precision 0.80. No validation
hits or recall are reported for a target without a train-selected threshold.

| Target precision | Configuration | Cosine threshold | NLI threshold | Validation hits | Validation precision | Validation recall |
|---:|---|---:|---:|---:|---:|---:|
| 0.80 / 0.90 / 0.95 | Cosine only | — | — | Unreachable | — | — |
| 0.80 / 0.90 / 0.95 | Cosine + Stage-0 | — | — | Unreachable | — | — |
| 0.80 / 0.90 / 0.95 | Cosine + Stage-0 + NLI (cosine 0.85) | — | — | Unreachable | — | — |
| 0.80 / 0.90 / 0.95 | Cosine + Stage-0 + NLI (cosine 0.95) | — | — | Unreachable | — | — |

**QQP:**

| Target precision | Configuration | Cosine threshold | NLI threshold | Validation hits | Validation precision | Validation recall |
|---:|---|---:|---:|---:|---:|---:|
| 0.80 | Cosine only | 0.93 | — | 7,377 | 0.8254 | 0.4091 |
| 0.80 | Cosine + Stage-0 | 0.92 | — | 8,212 | 0.8131 | 0.4486 |
| 0.80 | Cosine + Stage-0 + NLI | 0.85 | 0.23 | 8,109 | 0.8129 | 0.4429 |
| 0.80 | Cosine + Stage-0 + NLI | 0.95 | 0.01 | 4,356 | 0.8942 | 0.2617 |
| 0.90 | Cosine only | 0.98 | — | 1,899 | 0.9294 | 0.1186 |
| 0.90 | Cosine + Stage-0 | 0.98 | — | 1,861 | 0.9339 | 0.1168 |
| 0.90 | Cosine + Stage-0 + NLI | 0.85 | 0.99 | 1,691 | 0.9213 | 0.1047 |
| 0.90 | Cosine + Stage-0 + NLI | 0.95 | 0.97 | 2,526 | 0.9323 | 0.1582 |
| 0.95 | Cosine only | — | — | Unreachable | — | — |
| 0.95 | Cosine + Stage-0 | — | — | Unreachable | — | — |
| 0.95 | Cosine + Stage-0 + NLI | 0.85 | — | Unreachable | — | — |
| 0.95 | Cosine + Stage-0 + NLI | 0.95 | — | Unreachable | — | — |

### Stage-0 candidate populations

Unlike the full-split rejection totals above, these counts are restricted to
the specified cosine candidate set **before** Stage-0 filtering.

| Dataset | Population | Candidate pairs | Number mismatch: non-paraphrase / paraphrase | Negation mismatch: non-paraphrase / paraphrase |
|---|---|---:|---:|---:|
| PAWS | All validation | 8,000 | 95 / 52 | 49 / 13 |
| PAWS | Cosine >= 0.85 | 7,960 | 94 / 49 | 48 / 13 |
| QQP | All validation | 40,430 | 3,747 / 1,028 | 3,180 / 830 |
| QQP | Cosine >= 0.85 | 17,775 | 501 / 543 | 335 / 544 |

The CSVs include corresponding rows for every cosine threshold from 0.85
through 0.99. The counts for “All validation” intentionally preserve the
original diagnostic population; the cosine-filtered rows show the operational
candidate population before the guard rejects pairs.

### Insights and follow-up

1. No PAWS configuration met even 0.80 train precision within the tested
   thresholds. This is a grid-limited “unreachable” result, not evidence that
   no higher cosine cutoff could ever meet the target.
2. On QQP, 0.90 train precision is reachable, but validation recall remains
   low at those selected operating points. The 0.95 target was unreachable on
   the calibration grid for all configurations.
3. Increasing cosine from 0.85 to 0.99 sharply reduces recall on both datasets.
   The cosine-only and Stage-0 curves should be treated as candidate-retrieval
   comparisons, not safe cache-hit thresholds.
4. The Stage-0 guard rejects some positive examples; inspect the saved
   validation samples before changing its rules. Any future behavior change
   should be calibrated and reported with the same split discipline.

## 2026-11-07 — Banking77 frozen QQP operating points

### Protocol

- Dataset: Hugging Face `PolyAI/banking77`; 10,003 training examples were used
  as the candidate pool and 3,080 test examples as queries, following the
  existing Banking77 evaluator split handling.
- No Banking77 threshold selection was performed. All four settings were
  frozen from the QQP train-calibrated results:
  1. cosine + Stage-0, cosine 0.98;
  2. cosine + Stage-0 + bidirectional NLI, cosine 0.95 / NLI 0.97;
  3. reference cosine + Stage-0 + bidirectional NLI, cosine 0.85 / NLI 0.23;
  4. reference cosine + Stage-0, cosine 0.92.
- The candidate search considers the two most similar training examples. For
  NLI settings, the query is counted as an NLI call when at least one of those
  candidates passes cosine and Stage-0 and is sent to the reranker.
- `hits` counts any accepted cache result. A false positive is a hit whose
  returned training example has a different intent. Precision is same-intent
  hits divided by hits; recall is same-intent hits divided by all test queries.
  NLI call rate is NLI-called queries divided by test queries.
- Embedding model: `BAAI/bge-small-en-v1.5`. NLI model:
  `Xenova/nli-deberta-v3-small`, revision
  `6bc2a55c7c0f7e2bc68de60bb248e523e2612abb`.
- Run timestamp: `2026-11-07T09:45:22.648909+00:00`. Git HEAD recorded:
  `7c6177cdb4ff0a4f421db26f281bfbc57c21f3cc`; the worktree contained the
  evaluator changes that produced this run.
- Artifacts:
  [summary metrics CSV](../benchmarks/results/banking77_frozen_config_metrics.csv)
  and [top false-positive intent confusions CSV](../benchmarks/results/banking77_frozen_config_top_confusions.csv).

### Measured results

| Frozen config | Cosine | NLI | Hits | False positives | Same-intent hits | Precision | Recall | NLI-called queries | NLI call rate | Candidate pairs sent to NLI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cosine + Stage-0 | 0.98 | — | 744 | 11 | 733 | 0.9852 | 0.2380 | 0 | 0.00% | 0 |
| Stage-0 + NLI | 0.95 | 0.97 | 1,297 | 34 | 1,263 | 0.9738 | 0.4101 | 1,724 | 55.97% | 2,964 |
| Reference Stage-0 + NLI | 0.85 | 0.23 | 2,175 | 101 | 2,074 | 0.9536 | 0.6734 | 2,587 | 83.99% | 4,792 |
| Reference cosine + Stage-0 | 0.92 | — | 2,234 | 84 | 2,150 | 0.9624 | 0.6981 | 0 | 0.00% | 0 |

### Most frequent false-positive intent confusions

Each pair is `query intent → retrieved intent (false-positive count)`. The CSV
contains the same ranked lists and is the source for exact ordering; ties are
ordered by intent names.

| Frozen config | Top false-positive intent pairs |
|---|---|
| Cosine + Stage-0, 0.98 | `balance_not_updated_after_bank_transfer → pending_transfer (1)`; `card_arrival → card_delivery_estimate (1)`; `card_delivery_estimate → card_arrival (1)`; `declined_card_payment → declined_transfer (1)`; `get_disposable_virtual_card → disposable_card_limits (1)`; `pending_transfer → balance_not_updated_after_bank_transfer (1)`; `pin_blocked → change_pin (1)`; `top_up_by_cash_or_cheque → topping_up_by_card (1)`; `topping_up_by_card → transfer_into_account (1)`; `transfer_into_account → topping_up_by_card (1)`; `transfer_timing → pending_transfer (1)`. |
| Stage-0 + NLI, 0.95 / 0.97 | `card_arrival → card_delivery_estimate (3)`; `verify_my_identity → unable_to_verify_identity (3)`; `fiat_currency_support → exchange_via_app (2)`; `getting_virtual_card → get_disposable_virtual_card (2)`; `unable_to_verify_identity → verify_my_identity (2)`; `balance_not_updated_after_bank_transfer → pending_transfer (1)`; `card_delivery_estimate → card_arrival (1)`; `card_not_working → pin_blocked (1)`; `declined_card_payment → declined_transfer (1)`; `disposable_card_limits → get_disposable_virtual_card (1)`; `exchange_charge → exchange_rate (1)`; `exchange_via_app → fiat_currency_support (1)`; `get_disposable_virtual_card → disposable_card_limits (1)`; `pending_transfer → balance_not_updated_after_bank_transfer (1)`; `pending_transfer → pending_card_payment (1)`. |
| Reference Stage-0 + NLI, 0.85 / 0.23 | `pending_transfer → transfer_timing (3)`; `pin_blocked → change_pin (3)`; `transfer_fee_charged → top_up_by_bank_transfer_charge (3)`; `verify_my_identity → unable_to_verify_identity (3)`; `why_verify_identity → verify_my_identity (3)`; `balance_not_updated_after_bank_transfer → transfer_not_received_by_recipient (2)`; `balance_not_updated_after_bank_transfer → transfer_timing (2)`; `beneficiary_not_allowed → transfer_into_account (2)`; `card_arrival → card_delivery_estimate (2)`; `card_delivery_estimate → card_arrival (2)`; `declined_card_payment → declined_transfer (2)`; `exchange_charge → exchange_rate (2)`; `fiat_currency_support → exchange_via_app (2)`; `get_disposable_virtual_card → getting_virtual_card (2)`; `get_physical_card → change_pin (2)`. |
| Reference cosine + Stage-0, 0.92 | `exchange_via_app → fiat_currency_support (3)`; `pending_transfer → balance_not_updated_after_bank_transfer (3)`; `transfer_fee_charged → top_up_by_bank_transfer_charge (3)`; `verify_my_identity → unable_to_verify_identity (3)`; `why_verify_identity → verify_my_identity (3)`; `balance_not_updated_after_bank_transfer → transfer_timing (2)`; `card_arrival → card_delivery_estimate (2)`; `card_delivery_estimate → card_arrival (2)`; `declined_card_payment → declined_transfer (2)`; `exchange_charge → exchange_rate (2)`; `fiat_currency_support → exchange_via_app (2)`; `getting_virtual_card → get_disposable_virtual_card (2)`; `pin_blocked → change_pin (2)`; `request_refund → Refund_not_showing_up (2)`; `top_up_by_cash_or_cheque → topping_up_by_card (2)`. |

The 0.98 setting had the fewest false-positive hits, while 0.92 had the most
same-intent hits and highest query recall among these four settings. The
0.85/0.23 reference accepted more wrong-intent hits and invoked NLI on most
queries. These are Banking77 test measurements for frozen thresholds, not
threshold-selection evidence; no thresholds were changed based on them.

## Future result entries

For each new output, append another dated section with the exact command or
notebook, dataset revision/split, sample count, label definition, model versions,
thresholds, measured results, derived calculations, and any issues/actions.
Preserve previous entries so changes between runs can be compared honestly.
