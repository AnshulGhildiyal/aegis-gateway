# Phase 1 — Semantic cache summary

Phase 1 delivered the embedding-backed retrieval path, Stage-0 regex guard,
bidirectional NLI verifier, lookup pipeline, cached evaluation harnesses, and
validation/calibration reports. This closeout uses the saved result files only.
The Banking77 matched-precision output was refreshed on 2026-11-09 with
`--cached-scores-only`; no embedding or NLI model inference was performed.

## Matched-precision results

The table reports each saved training-selected operating point on its existing
evaluation split: Banking77 `test`; QQP and PAWS `validation`. Precision and
recall are evaluated values, not the training calibration values. PAWS and QQP
calibration targets are 0.80 and 0.90; Banking77 targets are 0.95, 0.97, and
0.98. For QQP Stage-0 + NLI, the table uses the higher-train-recall choice
between the saved cosine 0.85 and 0.95 calibration alternatives at each target.
The underlying files retain both alternatives. “Unreachable” means no
operating point on the saved calibration grid met the target.

| Dataset | Train precision target | Configuration | Selected cosine / NLI threshold | Evaluation split | Hits | Precision | Recall |
|---|---:|---|---|---|---:|---:|---:|
| Banking77 | 0.95 | Cosine only | 0.91 / — | test | 2,694 | 0.9529 | 0.8334 |
| Banking77 | 0.95 | Cosine + Stage-0 | 0.91 / — | test | 2,340 | 0.9568 | 0.7269 |
| Banking77 | 0.95 | Stage-0 + NLI | 0.85 / 0.06 | test | 2,294 | 0.9520 | 0.7091 |
| Banking77 | 0.97 | Cosine only | 0.95 / — | test | 1,901 | 0.9679 | 0.5974 |
| Banking77 | 0.97 | Cosine + Stage-0 | 0.95 / — | test | 1,724 | 0.9722 | 0.5442 |
| Banking77 | 0.97 | Stage-0 + NLI | 0.95 / 0.01 | test | 1,687 | 0.9727 | 0.5328 |
| Banking77 | 0.98 | Cosine only | 0.97 / — | test | 1,215 | 0.9745 | 0.3844 |
| Banking77 | 0.98 | Cosine + Stage-0 | 0.97 / — | test | 1,132 | 0.9770 | 0.3591 |
| Banking77 | 0.98 | Stage-0 + NLI | 0.95 / 0.99 | test | 945 | 0.9757 | 0.2994 |
| QQP | 0.80 | Cosine only | 0.93 / — | validation | 7,377 | 0.8254 | 0.4091 |
| QQP | 0.80 | Cosine + Stage-0 | 0.92 / — | validation | 8,212 | 0.8131 | 0.4486 |
| QQP | 0.80 | Stage-0 + NLI | 0.85 / 0.23 | validation | 8,109 | 0.8129 | 0.4429 |
| QQP | 0.90 | Cosine only | 0.98 / — | validation | 1,899 | 0.9294 | 0.1186 |
| QQP | 0.90 | Cosine + Stage-0 | 0.98 / — | validation | 1,861 | 0.9339 | 0.1168 |
| QQP | 0.90 | Stage-0 + NLI | 0.95 / 0.97 | validation | 2,526 | 0.9323 | 0.1582 |
| PAWS | 0.80, 0.90 | Cosine only | Unreachable | validation | — | — | — |
| PAWS | 0.80, 0.90 | Cosine + Stage-0 | Unreachable | validation | — | — | — |
| PAWS | 0.80, 0.90 | Stage-0 + NLI | Unreachable | validation | — | — | — |

Sources: [Banking77 matched-precision CSV](../benchmarks/results/banking77_loo_train_calibrated_test_matched_precision.csv),
[QQP matched-precision CSV](../benchmarks/results/paraphrase_ablation_qqp_train_calibrated_validation_validation_matched_precision.csv),
and [PAWS matched-precision CSV](../benchmarks/results/paraphrase_ablation_paws_train_calibrated_validation_validation_matched_precision.csv).
The table rounds saved precision and recall to four decimal places.

## Banking77 cosine-only safety dial

These are cosine-only thresholds selected on train leave-one-out data at each
target, then evaluated on the saved test split. The precision target is a
calibration constraint, not a guarantee of test precision.

| Train precision target | Cosine threshold | Test hits | Test precision | Test recall |
|---:|---:|---:|---:|---:|
| 0.95 | 0.91 | 2,694 | 0.9529 | 0.8334 |
| 0.97 | 0.95 | 1,901 | 0.9679 | 0.5974 |
| 0.98 | 0.97 | 1,215 | 0.9745 | 0.3844 |

Source: [Banking77 run JSON](../benchmarks/results/banking77_loo_train_calibrated_test.json)
and [cosine sweep CSV](../benchmarks/results/banking77_loo_train_calibrated_test_cosine_sweep.csv).

## Findings

- **Cosine-only provides the simplest usable baseline.** On Banking77, the
  train-selected settings met the 0.95 and 0.97 targets on test, while the
  nominal 0.98 target produced test precision below that target. As the
  threshold rose, recall fell.
- **Stage-0 has a measurable precision/recall trade-off, not a universal win.**
  On Banking77 at the 0.95 target it increased test precision versus
  cosine-only while lowering recall. On the QQP 0.80 point, it raised
  validation recall but produced slightly lower precision than cosine-only.
- **NLI did not show a consistent cross-dataset benefit.** It lowered
  Banking77 recall at each matched target; on QQP its result varied with the
  calibration choice. The prior QQP F1-based run also selected NLI threshold
  0.00, which passed every scored candidate and added no filtering.
- **Stage-0 and NLI are therefore OFF by default in production.** They remain
  explicitly enableable. A tenant can start with the configurable cosine
  threshold and enable either added check only after choosing an appropriate
  operating point for its data and false-hit risk.
- Production configuration is passed to `SemanticLookup`; for example:

  ```python
  lookup = SemanticLookup(
      cache,
      cosine_threshold=0.95,
      tenant_cosine_thresholds={"tenant-a": 0.92},
      stage0_enabled=False,
      nli_enabled=False,
  )
  ```

  Tenant-specific values override the default threshold and are range-checked.
  NLI must be supplied with a reranker when enabled.

## Limitations

- **PAWS targets are unreachable on the saved calibration grid.** This is a
  grid-limited result; it does not prove that no operating point exists outside
  the tested thresholds.
- **QQP labels are noisy/ambiguous.** The evaluation uses the dataset's binary
  paraphrase labels as-is; it does not independently relabel borderline pairs.
- **Cached measurements were pre-populated.** QQP and PAWS cosine scores were
  reused from the saved score caches. The Banking77 refresh validated and reused
  cached embeddings and NLI scores in strict cache-only mode. These reports do
  not represent a fresh model-inference run.
- **PAWS and QQP final test splits have not been run.** Their reported
  measurements are validation results. Banking77 test results are included
  because its saved LOO evaluator used test only after train-side threshold
  selection.

## Deferred work

- Stage-0 v2: improve handling of quantities, units, and negation without
  rejecting valid paraphrases.
- Build and label a targeted adversarial evaluation set for semantic-cache
  false hits.
- Complete manual labeling of an LMSYS-Chat sample and evaluate against those
  labels.
- Evaluate a fine-tuned verifier against the current pinned NLI model.
- Run the final PAWS and QQP test evaluation only after authorizing and freezing
  all thresholds.
