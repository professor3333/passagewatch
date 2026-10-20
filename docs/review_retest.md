# Same-Fish Question Re-test: Design

Release `passagewatch-0.3.3` added a question to the review page
([same-fish question](review.md#same-fish-question-on-the-review-page)). Before a reviewer counts
a track the release did not count, the page asks whether it is a different fish from the
counted tracks at overlapping times. Replaying recorded reviews shows that the question would
have appeared for 68 of the 70 count decisions that made counts worse. The replay cannot show
how reviewers answer it. This re-test measures that with new reviewers.

This page fixes the design, targets and analysis **before any reviewer starts**. Any later
change is recorded as a dated amendment. Results go in a separate report.

## Question and declared targets

**Does a reviewer using the page with the question leave the counts no worse than the
automatic ones, and stop counting pieces of already-counted fish?**

Two targets, judged per reviewer:

| Target | Measure | Met for one reviewer when |
|---|---|---|
| **1. No harm** (main) | Reviewed count error against the CFC reference, summed over all 21 clips, compared with the automatic counts' error on the same clips (10 passages) | reviewed error ≤ 10 |
| **2. Fewer fragment counts** | Fragment counts on set A (defined under Measures). R2 made 15 on the same clips with the same instructions and the page without the question; R1 made 5, under the earlier wording | at most 3 |

A target is **met** when it holds for at least two thirds of the reviewers: 2 of 3, or 3 of
4. With 3 or 4 reviewers these are descriptive thresholds, not statistical tests.

**If a target is missed,** the results report says so. The question's 10-frame window is
not tuned afterwards and re-tested on the same reviewers or clips. A changed question would
need a new declared design, new reviewers and new clips.

## Reviewers

- **3 or 4 people** who did not build PassageWatch and have never used its review page. That
  excludes the usability-study participants, R2 and the documentation-check reader. No
  fisheries background is needed.
- Codes `T1`–`T4`, assigned in the order people take part.
- Each reviews **alone**, without the developer present. Nobody tells them what the study is
  testing; they are told it checks the review page.
- The project owner records only each reviewer's code and the date. No names go in the
  repository.

## Materials

All clips come from the reviewed sample's 50 `kenai-holdout-v1` clips
([protocol](review_sample.md)). No detector trained on them and no setting was chosen on
them.

| Set | Clips | Recordings | Reference passages | Automatic error | Why |
|---|---|---|---|---|---|
| **A** | `s01`–`s15` | 12.4 min | 52 | 3 | The clips R1 and R2 reviewed without the question, so fragment counts compare directly |
| **B** | `s19`, `s20`, `s27`, `s28`, `s30`, `s34` | 5.4 min | 15 | 7 (all missed passages) | The first six clips of `s16`–`s50`, in code order, whose automatic counts are below the reference. They show the question's cost: a reviewer who answers "same fish" for a real separate fish keeps a missed fish missed |

- **One sequence.** The 21 clips are shown under new neutral codes `t01`–`t21`, in one order
  drawn with seed 0. The same order is used for every reviewer. Reviewers cannot tell set A
  from set B.
- **Release and page.** Release `passagewatch-0.3.3` (the same pipeline as 0.3.0–0.3.2). The
  review page is built from the commit recorded in the plan, with the logging change below.
  The frontend is the only difference from the page R2 used, apart from that logging.
- **Separate jobs.** Each reviewer gets their own job per clip, so nobody sees another's
  decisions.
- **Reference and automatic counts** are in the plan file. Reviewers never see them, and the
  page shows them only for the demo examples, which are not used here.

## Procedure

Each reviewer gets a link list and the instructions below, which are those R2 had
([protocol](review_sample.md#instructions-for-reviewers), including amendment 2, "one passage
per fish"). The only change is the clip codes. Nothing is said about the question. The page
shows it when it applies.

1. Open each link in order and play the recording. Use the speed control and frame steps as
   needed.
2. Give every track a decision, so that none shows "not reviewed":
   - Accept a correct passage. Keep an uncounted track uncounted when it really did not pass.
   - Use "Not a fish" for a counted track that isn't a fish or that duplicates another.
   - Change the direction of a passage counted the wrong way.
   - Count an uncounted track whose fish crossed and stayed across and is not already
     counted by another track.
   - Use Unresolved when unsure.
3. Add fish that crossed without a track. Check the audit windows.
4. Count each fish once. Take breaks freely; nothing is timed.

Expected time: about 50–70 minutes. A reviewer may stop and resume later.

## Measures

Computed from the review service by an analysis script (see Tooling). Only clips where every
track has a decision enter the figures. An incomplete clip is listed and left out.

| Measure | Per | Use |
|---|---|---|
| Reviewed and automatic count error against the reference (Σ\|R̂−R\|+\|L̂−L\|), and nMAE | reviewer, set A, set B, all | Target 1 |
| **Fragment counts:** count decisions on a track the release did not count, whose first and last boxes lie on the same side of the line, with a track the release counted overlapping it in time or within 10 frames (`describe_changes` in `passagewatch.evaluation.review_sample`, the definition behind R2's 15) | reviewer, set A | Target 2 |
| The same, on set B | reviewer | reported |
| **Answers to the question:** "Different fish" and "Same fish" (amendment 1) | reviewer, set | reported |
| Missed passages recovered on set B: how much of the automatic counts' shortfall the reviewed counts close | reviewer | the question's cost, reported |
| Every changed track, described as in the reviewed sample (where it starts and ends, counted tracks near it) | track | reported |
| Time from first to last decision, per clip and in total | reviewer | reported |

The reference is not ground truth. A reviewed count that differs from it is listed, not
resolved.

## Analysis

- **Per reviewer,** both targets and every measure above, with each reviewer's clips listed.
- **Against R1 and R2** on set A: fragment counts and reviewed error. These are historical
  comparisons: different people, on a different day, R2 with the same instructions, R1 with
  the earlier wording. They are context, not a controlled comparison.
- **No pooling into a significance test.** With 3 or 4 reviewers the results are descriptive.
  Each reviewer's numbers are reported, as is whether each target was met under the rule
  above.

## Threats to validity

- **Few reviewers and clips.** Every figure is rough. One careless or unusually careful
  reviewer changes a target's outcome.
- **No concurrent control.** Nobody uses the page without the question in this study.
  Comparing with R1 and R2 mixes the question's effect with differences between people and
  sessions. A crossover (each reviewer with and without the question) was rejected: after
  seeing the question, a reviewer is likely to look for fragments on the page without it too.
- **The 10-frame window was chosen after seeing earlier reviews.** The clips in set A are
  ones it was chosen on, which favours target 2. Set B was not used to choose it.
- **The reference is not ground truth.** Target 1 uses it. The automatic counts on set A are
  already close to it (error 3 of 52), so there is little room to improve there and much to
  worsen.
- **Instructions mention fragments.** "One passage per fish" already tells reviewers about
  split tracks, as it told R2, who still made 15 fragment counts. The question is the change
  under test, not the instructions.
- **One river, two cameras, one season**, as for the holdout.

## Tooling

- **Answers to the question are logged.** The page sends a `reason` with "Same fish: keep
  uncounted" and "Different fish: count" (`SAME_FISH_REASONS` in `frontend/src/review.ts` and
  `passagewatch.evaluation.same_fish`). API and counts are unchanged.
- **`scripts/retest_service.sh start|status|stop`** runs the re-test's service on this
  machine: data in `var/retest/` (not committed), port 8012, release `passagewatch-0.3.3`.
- **`scripts/prepare_retest.py`** (run once, from a clean checkout) writes `retest/plan.json`
  with the clips, codes, sets, order, reference and automatic counts, the release and the
  commit whose review page is served. It creates each reviewer's jobs and checks that the
  automatic counts equal the reviewed sample's. It writes each reviewer's link list
  `retest/T<n>.md`, with the instructions under Procedure. It refuses to run if the plan
  exists.
- **`scripts/analyze_retest.py`** computes the measures and both targets into
  `retest/results.json` (`passagewatch.evaluation.retest`). It also gives R1's and R2's
  set-A figures from the reviewed sample: fragment counts 5 and 15, reviewed errors 11 and
  18.

## Amendments

### Amendment 1 (2026-10-20, before any reviewer)

- **Not recorded: when the question was shown, and "Back".** Recording them would need a
  new kind of event in the service, because review events change the result and these must
  not. Only the two answers are recorded, so a question closed with "Back" leaves no trace.
- **The reviewers' instructions** are the protocol's (`docs/review_sample.md`, "Instructions
  for reviewers" with "One passage per fish") with three changes: an opening paragraph, no
  line about starting the service, and the new clip codes. R2 also had a one-line reminder
  to count each fish once in their link list. Here, "One passage per fish" is part of the
  instructions in each link list.

## Status

Declared 2026-10-20. Tooling built. No reviewer has started; the plan has not been prepared.
