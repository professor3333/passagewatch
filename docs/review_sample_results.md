# Reviewed Development Sample (Stage 9): Results

Results of the protocol in [review_sample.md](review_sample.md), as amended before any review
(both reviewers review `s01`–`s15`) and once more after R1 finished (amendment 2: "one
passage per fish" made explicit for R2). Numbers come from `review_sample/results.json`,
written by `scripts/analyze_review_sample.py` from the review service on 2026-10-17. The
results are descriptive; no targets were set.

## Summary

- **Both reviewers finished all 15 clips:** every track has a decision and every audit window
  is checked. Release `passagewatch-0.3.1` (same pipeline as 0.3.0), kenai-holdout-v1 clips,
  12.4 minutes of recordings, 98 tracks, 52 reference passages.
- **Reviewing made the counts differ more from the CFC reference, not less.** nMAE against
  the reference: automatic 0.058, R1 reviewed 0.212, R2 reviewed 0.346.
- **Nearly every change was the same kind:** counting a track the release had not counted.
  Of 23 changes (R1 7, R2 16), 21 count a track whose own first and last boxes lie on the
  same side of the line, and 22 are on tracks that overlap in time, or nearly, with a track
  the release already counted. The other two are the same redirected track (s08 #3), changed
  by both reviewers. Neither reviewer removed a counted track, marked a track unresolved, or added a
  missed fish.
- **The reviewers agree on 87 of 98 tracks (Cohen's kappa 0.78)** and give the same counts on
  9 of 15 clips. All 11 disagreements are uncounted tracks that R2 counted and R1 did not, or
  the reverse (one).
- **These labels do not do what the protocol intended**, which was to provide review-quality
  labels independent of the reference. They do show where human review in the current
  interface goes wrong, the same failure the usability study found
  ([usability_results.md](usability_results.md)).

## Sample and reviewers

| | |
|---|---|
| Clips | `s01`–`s15`, a random subset of the 50-clip stratified sample (codes are in random order) |
| Strata covered | LeftFar: 0 passages 1, 1–2 passages 7, 3–4 passages 1, 5+ passages 1; LeftNear: 1–2 passages 1, 3–4 passages 0, 5+ passages 4 |
| R1 | The project owner. Reviewed under the original wording |
| R2 | Someone who did not build PassageWatch (the owner confirms they reviewed alone and watched each clip). Reviewed under amendment 2 |
| Interface | The review page of release 0.3.1, with the panel from PR #52 ("Keep uncounted" / "Count as passage" for uncounted tracks) |

## Counts against the CFC reference

Right / left passages per clip. The reference is not ground truth; a difference is listed,
not resolved.

| Clip | Reference | Automatic | R1 | R2 |
|---|---|---|---|---|
| s01 | 11 / 0 | 10 / 0 | 10 / 0 | 10 / 0 |
| s02 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |
| s03 | 0 / 2 | 0 / 2 | 0 / 2 | 0 / 4 |
| s04 | 1 / 0 | 1 / 0 | 2 / 0 | 1 / 0 |
| s05 | 5 / 0 | 5 / 0 | 5 / 0 | 6 / 0 |
| s06 | 5 / 0 | 5 / 0 | 5 / 0 | 6 / 0 |
| s07 | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 |
| s08 | 0 / 1 | 1 / 1 | 4 / 0 | 4 / 0 |
| s09 | 1 / 0 | 1 / 0 | 2 / 0 | 2 / 0 |
| s10 | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 |
| s11 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| s12 | 9 / 0 | 8 / 0 | 8 / 0 | 10 / 0 |
| s13 | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 |
| s14 | 3 / 0 | 3 / 0 | 5 / 0 | 5 / 0 |
| s15 | 8 / 0 | 8 / 0 | 8 / 0 | 12 / 0 |
| **Total error** | | **3** | **11** | **18** |
| **nMAE** (52 passages) | | **0.058** | **0.212** | **0.346** |

On these 15 clips the release's automatic counts differ from the reference by 3 passages
(two fish missed in s01 and s12, one extra right passage in s08). Neither reviewer found the
two missed fish, in the clips or in the audit windows.

## What the reviewers changed

| | R1 | R2 |
|---|---|---|
| Tracks reviewed | 98 | 98 |
| Kept as the release decided | 91 | 82 |
| Counted (uncounted → passage) | 6 | 15 |
| Redirected | 1 | 1 |
| Removed | 0 | 0 |
| Unresolved | 0 | 0 |
| Missed fish added | 0 | 0 |

Each changed track, described by the analysis (`changes` in `results.json`): where its first
and last boxes lie (x as a fraction of the width; the line is at 0.5), whether it crosses the
line on its own by the counting policy's start/end-side rule, and the tracks the release
counted that overlap it in time or lie within 10 frames of it.

| Clip, track | Change | x start → end | Crosses alone | Counted tracks near it | By |
|---|---|---|---|---|---|
| s03 #2 | counted ← | 0.79 → 0.69 | no | #3, #4 | R2 |
| s03 #5 | counted ← | 0.11 → 0.14 | no | #4 | R2 |
| s04 #1 | counted → | 0.08 → 0.07 | no | none (#2 starts 50 frames later) | R1 |
| s05 #5 | counted → | 0.11 → 0.32 | no | #2, #3, #4, #6, #7 | R2 |
| s06 #2 | counted → | 0.03 → 0.07 | no | #3–#6 | R2 |
| s08 #1 | counted → | 0.82 → 0.56 | no | #3, #4 | R1, R2 |
| s08 #2 | counted → | 0.60 → 0.57 | no | #3 | R1, R2 |
| s08 #3 | ← changed to → | 0.50 → 0.45 | yes (leftward) | #4 | R1, R2 |
| s09 #2 | counted → | 0.61 → 0.89 | no | #1 (ends at 0.61, 6 frames earlier) | R1, R2 |
| s12 #7 | counted → | 0.69 → 0.68 | no | 8 tracks | R2 |
| s12 #12 | counted → | 0.68 → 0.89 | no | 8 tracks | R2 |
| s14 #3 | counted → | 0.43 → 0.40 | no | #1, #2, #4 | R1, R2 |
| s14 #5 | counted → | 0.57 → 0.69 | no | #1, #2, #4 | R1, R2 |
| s15 #5 | counted → | 0.13 → 0.38 | no | 7 tracks | R2 |
| s15 #8 | counted → | 0.03 → 0.21 | no | 7 tracks | R2 |
| s15 #14 | counted → | 0.65 → 0.86 | no | 7 tracks | R2 |
| s15 #15 | counted → | 0.94 → 0.90 | no | 5 tracks | R2 |

A track that does not cross on its own can still be part of a fish that crossed, when the
tracker split the fish into pieces. Counting such a piece is right only if no other track of
the same fish is counted. The analysis cannot tell which counted track, if any, belongs to
the same fish, so this table does not say any change was wrong. Two things are clear from
it: the changes are almost all of one kind, and in the clearest cases (s09 #2 continuing
where the counted #1 ends; s04 #1 at the image edge before the counted #2 appears) the
change counts a second piece of a fish the release already counted. In dense clips (s12,
s15) "near" means little, because many tracks overlap.

### By triage state

| Triage | Tracks | Changed by R1 | Changed by R2 |
|---|---|---|---|
| suggested | 93 | 6 (6%) | 15 (16%) |
| needs review | 1 | 0 | 0 |
| unresolved | 4 | 1 | 1 |

Almost all changes are on `suggested` tracks, the state the review score treats as least in
need of review. Because the changes themselves are doubtful, this says little about the
triage.

## Agreement between R1 and R2

| | |
|---|---|
| Tracks both decided | 98 |
| Same outcome (counted →, counted ←, not counted, unresolved) | 87 (89%) |
| Cohen's kappa | 0.78 |
| Clips with the same counts | 9 of 15 |

The 11 disagreements: R2 counted s03 #2, #5; s05 #5; s06 #2; s12 #7, #12; s15 #5, #8, #14,
#15 (R1 kept them uncounted); R1 counted s04 #1 (R2 kept it uncounted). They agree on all
four s08 decisions and on s09 and s14. The reviewers had different instructions for exactly
this kind of decision (amendment 2), so the agreement is not purely two readings of the
same footage. R2, who had the explicit "one passage per fish" rule, counted more uncounted
tracks than R1.

## Timing

From the review events' timestamps (2026-10-17, UTC):

| | First decision | Last decision | Span | Recordings |
|---|---|---|---|---|
| R1 | 09:13 | 09:50 | 37.2 min (includes finishing one track and three clips' audit windows at the end) | 12.4 min |
| R2 | 09:59 | 10:11 | 11.6 min | 12.4 min |

R2's span is shorter than the recordings' total length. The interface has a speed control,
and a reviewer may watch a clip before making the first decision, so the span does not show
how long each clip was watched. The owner confirms R2 watched each clip.

## Interpretation

- **The main finding is about the interface, not the model.** In the usability study, 36 of
  38 direction changes turned non-passing tracks into passages. PR #52 replaced "Set → right"
  on every track with "Keep uncounted" / "Count as passage" on uncounted tracks only. In this
  sample, with the new panel, 21 of 21 "count" decisions by two reviewers (one with an
  explicit one-passage-per-fish rule) went to tracks that do not cross on their own, and both
  reviewers' counts moved away from the reference. The panel change did not stop it.
- **A likely mechanism.** The tracker splits some fish into pieces. A reviewer who watches a
  fish cross sees a piece of it in the track list, uncounted, and counts it, without seeing
  that another piece of the same fish is already counted. The review page shows one track at
  a time and gives no sign of which tracks may be the same fish.
- **What the labels can and cannot be used for.** They cannot be used as independent
  review-quality labels for the release's counts, which was the protocol's aim. They should
  not be used as training data. They are evidence that, in this interface, an unaided
  reviewer can make good automatic counts worse.
- **What would address it** (each needs its own measured check):
  showing tracks that overlap in time and space as a group, with the group's count; asking
  "Is this fish already counted by another track?" before "Count as passage"; and, in the
  review guidance, putting the counted tracks of the same moment next to the track being
  decided. *Added later:* the review page now lists the counted tracks at overlapping times
  and asks whether the track is a different fish before counting it
  ([same-fish question](review.md#same-fish-question-on-the-review-page)). Its effect is not
  measured.

## Threats to validity

- **15 clips and two reviewers.** Every figure here is rough.
- **R1 built the system**, and saw no reference counts until their review was complete.
  R1's intent behind the counted tracks (a piece of a fish that crossed, or separate fish)
  was not recorded.
- **Different instructions** for R1 and R2 (amendment 2).
- **The reference is not ground truth.** Reviewed counts that differ from it are listed, not
  resolved. The analysis describes where changed tracks lie; it does not decide whether any
  fish was counted twice.
- **No closing questions** were asked of either reviewer, so their reasons are not known.

## Files

- `review_sample/results.json`: per clip and reviewer, counts, verdicts, changed tracks with
  positions, timing; agreement between reviewers.
- `review_sample/plan.json`: the sample, reference and automatic counts, assignments and both
  amendments.
- Reproduce, with the review service running on its data (`var/review-sample/`, not
  committed): `uv run python scripts/analyze_review_sample.py --url http://127.0.0.1:8011`.
