# Reviewed Development Sample (Stage 9): Protocol

The design asks for review-quality labels made by a person, not taken from the CFC
annotations (`docs/design.md`, "Additional data you can realistically create yourself"):
a sample of development clips in which every count contribution the release proposes is
marked right or wrong, missed fish, duplicates, wrong directions and ambiguous cases are
recorded, a second reviewer checks part of it, and disagreements are kept. This page fixes
the protocol **before any clip is reviewed**. Results go in a separate report.

## What it is for

- **Labels independent of the reference.** The review score and triage
  ([review](review.md)) were measured against CFC's annotations. Here a person judges the
  same kind of output directly.
- **How reliable those labels are.** Two reviewers on the same clips give an agreement
  figure, so the labels are not taken as ground truth.
- **Where the reference may be wrong.** A reviewed count that differs from CFC's is listed
  as a disagreement. Nobody decides afterwards who was right.

No targets are set. The results are descriptive.

## Sample

By `scripts/prepare_review_sample.py` (seed 0, run once; `passagewatch.evaluation.review_sample`):

- **Pool:** the 174 `kenai-holdout-v1` clips, which no detector trained on and no setting was
  chosen on. The release's automatic counts come from its holdout report. Excluded: the 14
  usability-study clips, the demo examples, and any clip named in the documentation. That
  leaves 158.
- **Strata:** camera (LeftFar, LeftNear) × reference passages (0, 1–2, 3–4, 5 or more).
- **50 clips**, proportional to the pool per stratum, at least one per stratum, about 41
  minutes of recordings in all. They get neutral codes `s01`–`s50` in a random order.
- **Reviewer R1** (the project owner) reviews all 50 (amended to `s01`–`s15`, below).
- **Reviewer R2** (someone who did not build PassageWatch) reviews `s01`–`s15`, a random
  subset because the codes are in random order.
- **Separate jobs.** Each reviewer gets their own job per clip, a result-cache hit of the
  release's analysis, so neither sees the other's decisions.
- **Reference counts** are in `review_sample/plan.json`. Reviewers do not open it until
  they have finished.

### Amendment (2026-10-13, before any review)

The project owner does not have time for 50 clips (about 2.5 hours of reviewing). **Both
reviewers review the same 15 clips, `s01`–`s15`** (about 12 minutes of recordings), so the
R1–R2 comparison is unchanged and covers every reviewed clip. `s16`–`s50` are not reviewed;
their jobs stay in the service unused. The amendment is recorded in `review_sample/plan.json`
(`amendments`), and the analysis follows its assignments. Because the codes are in random
order, the 15 clips are a random subset of the stratified sample, so a stratum may be thin
or missing; the results report gives the strata the 15 clips cover. Figures from 15 clips
are descriptive and rough.

## Instructions for reviewers

You need the service running (`scripts/review_sample_service.sh start`) and your link list
(`review_sample/R1.md` or `R2.md`). For each clip:

1. Open the link and play the recording. Use the speed control and frame steps as needed.
2. Give **every track** in the list a decision. A clip is not complete until no track
   shows "not reviewed".
   - **Accept** a counted passage that is right, fish and direction; **Keep uncounted** an
     uncounted track that really did not pass.
   - **Not a fish: don't count** for a counted track that is not a fish, or that duplicates
     another track of the same fish.
   - **Change to** the other direction for a passage counted the wrong way.
   - **Count as passage → / ←** for an uncounted track whose fish did cross the line and
     stay across, **and is not already counted by another track** (amendment 2, below).
   - **Unresolved** when you cannot tell. This is a valid answer; do not guess.
3. Add every fish that crossed but has no track: **Add missed fish here** at the frame where
   it crosses. Check the **random audit** windows and mark them checked.
4. Tick the clip off in your list. Take breaks whenever you like; nothing is timed.

A fish that crosses and comes back does not count. Directions are as seen in the image
(→ right, ← left).

**One passage per fish** (amendment 2). The tracker sometimes splits one fish into several
tracks, for example one piece before the line and another after it, or puts two boxes on the
same fish. Such a fish counts **once**:

- If one of its tracks is already counted, keep its other tracks uncounted (**Keep
  uncounted**), or remove a second counted one (**Not a fish: don't count**).
- If none of its tracks is counted but the fish did cross and stay across, count it on
  **one** track (**Count as passage**) and keep the others uncounted.
- Before counting an uncounted track, look for another track on the same fish at the same
  time or just before and after it.

### Amendment 2 (2026-10-17, after R1 finished, before R2 started)

The instruction for **Count as passage** said "an uncounted track whose fish did cross the
line". Read literally, that covers every track of a fish that crossed, including pieces of a
fish another track already counts. The rule **one passage per fish** above makes the intended
meaning explicit. It applies to R2. R1 reviewed under the earlier wording, and R1's labels
are kept unchanged. The results report says which wording each reviewer had, so the R1–R2
comparison partly reflects the change of instructions. The assignments are unchanged
(`review_sample/plan.json`, `amendments`).

## Measures

Computed by `scripts/analyze_review_sample.py` from the service. Only complete clips enter
the figures.

| Measure | Per |
|---|---|
| Each track's verdict: kept, removed, counted, redirected, unresolved | track, reviewer |
| Passages added by the reviewer | clip, reviewer |
| Reviewed and automatic counts against the CFC reference (nMAE) | reviewer |
| Clips whose reviewed counts differ from the reference (listed, not resolved) | reviewer |
| Share of reviewed tracks the reviewer changed, per triage state | reviewer |
| R1–R2: share of tracks with the same outcome, Cohen's kappa, every disagreement | the 15 shared clips |

## Threats to validity

- **R1 built the system.** They know how it fails, which may make them look harder at some
  tracks than a technician would. R2's agreement with R1 bounds this, but does not remove it.
- **The interface shapes the labels.** Reviewers judge the release's tracks. A fish with no
  track is found only by watching, and only if the reviewer adds it.
- **One river, two cameras, one season.** The same limits as the holdout.
- **The reference is not ground truth.** Disagreements with it are reported both ways.
- **The reviewers had different instructions** for counting a track the release did not count
  (amendment 2). Agreement between them is not purely agreement between two readings of the
  same footage.
