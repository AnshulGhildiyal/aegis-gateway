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

## Future result entries

For each new output, append another dated section with the exact command or
notebook, dataset revision/split, sample count, label definition, model versions,
thresholds, measured results, derived calculations, and any issues/actions.
Preserve previous entries so changes between runs can be compared honestly.
