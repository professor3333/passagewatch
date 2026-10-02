# Review Scores, Triage and Audits (Stage 9)

PassageWatch suggests counts; a technician confirms them. This page describes how the system
decides where the technician should look first, and how that is measured. Everything here
is **heuristic and provisional**: thresholds are chosen in Stage 9, after the detector for
the next release is fixed.

## Review score (`review-score-v0`)

`passagewatch.calibration.review_score` scores every trajectory from six pieces of evidence,
each mapped to `[0, 1]` (1 = strong) with fixed anchors:

| Component | Feature | 0 at | 1 at |
|---|---|---|---|
| detection | mean detector score | 0.2 | 0.6 |
| duration | observations | 3 | 15 |
| continuity | gap fraction (`1 − observations / span`) | 0.5 | 0 |
| motion | straightness: net displacement / path length, in meters | 0 | 1 |
| separation | share of frames with no other trajectory's box overlapping (IoU > 0.1) | 0 | 1 |
| line distance | normalized x distance of the nearer endpoint from the counting line | 0 | 0.15 |

The review score is their geometric mean, with each component floored at 0.01. A
geometric mean lets one weak component (a short track, or weak detections) lower the score
sharply instead of being averaged away. **It is a ranking heuristic, not a probability.**
The product calls it a "review score" and shows no count confidence intervals until a
calibrator has been fitted on separate training groups and validated.

## Triage

| State | Rule (provisional) |
|---|---|
| `unresolved` | a passage that goes mostly back and forth (straightness < 0.3), or that starts or ends within 0.02 of the line |
| `needs_review` | a passage with review score < 0.6, or a non-passing track of at least 5 observations that starts or ends within 0.05 of the line (it may be a fragment of a missed passage) |
| `suggested` | everything else |

Triage never changes the automatic counts, which come only from the counting policy. A
reviewer's decisions are separate, append-only review events.

## Calibration version `review-v0`

A release bundle declares `calibration_version: review-v0` to switch all of this on:
`review-score-v0` with the thresholds above, and `audit-v0` windows of 5 s covering at least
10% of the unflagged frames, kept 1 s away from flagged trajectories
(`passagewatch.calibration.versions`). The version is part of the pipeline's config hash, so
it is part of every result's cache key and every export. The worker computes scores,
triage and audit windows with the predictions, and they are stored with them, never
recomputed. A bundle without a calibration version (such as `passagewatch-0.2.0`) gives
tracks without scores and jobs without audit windows.

In the API, `GET /v1/jobs/{id}/tracks?order=queue` lists tracks in review-queue order, and
`GET /v1/jobs/{id}/audit` lists the audit windows with their state. A reviewer marks a window
with the review action `mark_audited`, and records a fish found there with `add_passage` and
its `audit_window`. Exports (report version 2) add each track's triage and review score, and
the audit windows, how many were checked, and the passages found in them.

## Random audits (`audit-v0`)

A queue built from predicted trajectories cannot show a fish that was never tracked. So each
recording also gets randomly placed windows of **unflagged** footage: footage away from
every counted or flagged trajectory (`passagewatch.calibration.audit`). The windows are
drawn reproducibly from the job's ID, never overlap, and cover at least a set share of the
unflagged frames. A reviewer watches them and adds any passage the system missed.

## How prioritization is measured

`scripts/evaluate_review.py` uses the one-to-one error analysis
(`passagewatch.evaluation.errors`) as ground truth on a development partition:

- Reviewing a trajectory costs 3 s plus its duration, and **reveals** its own false count
  (a duplicate, background, non-passing-fish or wrong-direction passage), plus every missed
  reference passage it follows (split, merged, partial or ambiguous cases).
- A reference passage with no trajectory at all can only be found by an audit. It is
  *unreachable* from the queue.
- The curve is the share of errors found against the share of total review time spent, for
  the queue order (triage, then passages before other tracks, then ascending review score),
  for the review score alone, and for random order (averaged over 200 orders).

The design's planning target is 80% of errors found within 40% of the review time.

## First measurement (development data, provisional)

The released YOLOX-Tiny (epoch 25, threshold 0.2) on **kenai-val** (64 clips). Its 30
counting errors match the error analysis: 25 are reachable from the queue (344
trajectories, about 44 minutes of review), and 5 are fish that nothing tracked. The
`review-score-v0` anchors and thresholds were set before this run.

Share of **reachable** errors found:

| Review time | 10% | 20% | 30% | 40% | 60% | 100% |
|---|---|---|---|---|---|---|
| Queue (triage, then score) | 0.52 | 0.72 | 0.80 | **0.84** | 0.84 | 1.00 |
| Review score alone | 0.12 | 0.44 | 0.72 | 0.84 | 0.96 | 1.00 |
| Random order | 0.16 | 0.30 | 0.43 | 0.53 | 0.72 | 1.00 |

Share of **all** errors found at 40% of the review time: 0.70 for the queue and 0.44 for
random order. Without audits the ceiling is 0.83 (25 of 30).

| Triage | Trajectories | Leading to an error |
|---|---:|---:|
| `unresolved` | 4 | 0 |
| `needs_review` | 15 | 9 |
| `suggested` | 325 | 34 |

Several trajectories can lead to the same error, so the last column sums to more than 25.

With `review-v0`'s audit settings (windows of 5 s covering at least 10% of the unflagged
frames, 1 s away from flagged trajectories), audits covering 12.2% of all frames showed 2 of
the 5 fish that nothing tracked.

What this shows, with the caution that 64 clips and 30 errors are a small sample:

- **The queue concentrates errors early.** At 30–40% of the review time it finds 80–84% of
  the reachable errors, against 43–53% for random order.
- **`needs_review` is useful:** 9 of its 15 trajectories lead to an error, against about 12%
  of all trajectories.
- **The `unresolved` rule is not.** Its 4 trajectories were all correct. Back-and-forth paths
  and endpoints near the line are not, on this data, signs of a wrong count. The rule needs
  rethinking before release.
- **Audits are needed** to approach the 80%-of-all-errors target, because a sixth of the
  errors are fish the queue cannot reach.

Next: once the detector is fixed, choose the thresholds on kenai-val and confirm them once on
`kenai-holdout-v1`.
