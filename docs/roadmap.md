# Roadmap

PassageWatch is built in twelve stages grouped into six milestones. Each stage has a
concrete completion test, and a stage counts as done only when its test passes. The
official CFC **test locations are used only in Stage 11**; every earlier stage uses the
training and validation data from Kenai.

| Milestone | Stages | Result | Tag |
|---|---|---|---|
| Trustworthy foundation | 1–3 | Validated data, counting logic, classical baseline | `v0.1.0` |
| Working neural pipeline | 4–5 | Reproducible detection, tracking, and counting | `v0.2.0` |
| Usable MVP | 6–7 | Upload, analysis, human correction, export | `v0.3.0` |
| Stronger decision support | 8–9 | Evidence-based improvements and uncertainty handling | `v0.4.0` |
| Reliable deployment | 10–11 | Tested, optimized, evaluated, and monitored application | `v1.0.0` |
| Validated usefulness | 12 | Measured review study and reproducible documentation | — |

## Stages

- [x] **1. Scope and repository** — package, lockfile, CI, [scope](scope.md), and
  [counting policy](counting_policy.md).
  *Done when* a fresh installation runs the package and its tests.
- [ ] **2. Dataset pipeline** — reproducible download with checksums, parsing, coordinate
  conversion, validation report, annotation viewer, versioned manifest that preserves the
  official splits, and a dataset card.
  *Done when* a sequence replays with correctly aligned annotations and invalid samples
  produce clear validation errors.
- [ ] **3. Classical baseline and counting evaluator** — background subtraction,
  connected components, size and motion filtering, association, directional counting,
  and a benchmark-compatible nMAE evaluator checked against the official results.
  *Done when* one command turns a sequence into tracks, directional counts, and an
  evaluation report.
- [ ] **4. First neural detector** — fine-tune a compact COCO-pretrained detector on a
  small training subset, with configuration files, checkpoints, resume, and MLflow tracking.
  *Done when* training can be reproduced and resumed, and it detects fish usefully on unseen
  validation sequences.
- [ ] **5. Detection, tracking, and counting** — neural detections through the tracker,
  a comparison with the classical pipeline, ByteTrack as a separate experiment, and
  timestamped evidence with full provenance.
  *Done when* the effect of detection and tracking errors on counts is explained and measured.
- [ ] **6. Backend and asynchronous processing** — FastAPI, persistent jobs in SQLite, a
  separate worker, validation, limits, timeouts, idempotency, and a container image.
  *Done when* a client can upload, start analysis, poll, and retrieve results without a
  long-held request.
- [ ] **7. Review interface (MVP)** — video with trajectories and the counting line,
  counts and unresolved cases, jump-to-evidence, accept/reject/correct, preserved
  revisions, and CSV/JSON export.
  *Done when* someone can upload, inspect, correct, and export reviewed counts.
- [ ] **8. Error analysis and controlled improvements** — categorized development-set
  failures, a single-frame vs. temporal-input comparison, tracker tuning on fixed
  detections, and slice evaluation.
  *Done when* every retained change has a measured benefit or a documented engineering reason.
- [ ] **9. Uncertainty and review prioritization** — review-score rules, ranking,
  explicit unresolved cases, an independently reviewed development sample, calibration only
  if the data supports it, and random audits of unflagged footage.
  *Done when* strong and uncertain results are distinguishable, and the efficiency of
  prioritization is measured.
- [ ] **10. Hardening and optimization** — failure and recovery tests, a preprocessing
  parity test, model-regression tests, profiling, ONNX Runtime and quantization where
  profiling justifies them, Docker Compose, and immutable inference bundles.
  *Done when* the system recovers from expected failures without producing duplicate
  results and stays within its documented resource limits.
- [ ] **11. Final evaluation and deployment** — freeze the pipeline, evaluate the
  deployment artifact on the held-out test locations with confidence intervals, measure
  runtime, and deploy with monitoring and a tested rollback.
  *Done when* the deployed system is the same as the evaluated release, and its
  limitations are documented.
- [ ] **12. Usefulness study and documentation** — a counterbalanced review-time study,
  three demo examples (clear, difficult, unfamiliar camera), and a README, model card,
  dataset card, and architecture guide verified by another person.
  *Done when* the important work can be understood quickly and reproduced independently.

## Out of scope for Version 1

Species classification, raw sonar-format and hardware integration, continuous streaming,
and distributed infrastructure. See [scope.md](scope.md).
