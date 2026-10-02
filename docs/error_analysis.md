# Error Analysis (Stage 8, development data)

Why counts are wrong, trajectory by trajectory, on the 64 kenai-val clips (183 true
passages: 175 rightward, 8 leftward). The test locations are not used. Produced by
`scripts/analyze_errors.py` (`passagewatch.evaluation.errors`). The JSON reports, with every
error case and its clip and frames, are written to `runs/errors/` and not committed.

## Method

Predicted and reference trajectories are matched when they share at least 3 frames in which
their boxes overlap with IoU ≥ 0.3. Passages are attributed **one to one**: each predicted
passage goes to the same-direction reference passage it shares the most frames with, and
only one predicted passage can count a reference passage.

| Reference passage outcome | Meaning |
|---|---|
| counted | Counted once, in the right direction |
| wrong_direction | A matching track counts the opposite direction |
| missed_fish | No predicted track follows the fish at all |
| merged_track | Its only track also follows another fish and counts that one instead (or neither) |
| split_track | Several fragments follow it, and none spans the line |
| ambiguous_start_end | One track follows it but starts or ends within 0.05 of the line |
| partial_track | One track follows it but covers too little of its path to cross |

| Predicted passage outcome | Meaning |
|---|---|
| counted | Counts a real passage (one to one) |
| duplicate | Counts a fish that another track already counts |
| wrong_direction | Follows a passing fish but counts the other direction |
| background | Follows no fish: clutter or debris |
| non_passing_fish | Follows a real fish that does not pass |

The project's "corrupted timing/config" category cannot occur on CFC clips, which have fixed
frame rates and configurations. It applies to uploads only.

## Results

| | YOLOX-Tiny release (`passagewatch-0.2.0`) | `classical-v2` |
|---|---:|---:|
| Counting nMAE | 0.120 | 0.235 |
| **Reference passages counted** | **162 / 183** | 132 / 183 |
| merged_track | 6 | 14 |
| missed_fish | 5 | 19 |
| split_track | 5 | 6 |
| partial_track | 3 | 8 |
| ambiguous_start_end | 2 | 4 |
| wrong_direction | 0 | 0 |
| **False predicted passages** | **9** | 34 |
| duplicate | 7 | 16 |
| background | 1 | 17 |
| non_passing_fish | 1 | 1 |
| **Passage-level errors** | **30** | 85 |

**Direction confusion** (the release; rows are reference, columns predicted; each reference
trajectory is paired with its best-matching predicted track, and unmatched predicted
passages are added in the "none" row):

| | → right | ← left | none |
|---|---:|---:|---:|
| reference → right | 151 | 0 | 24 |
| reference ← left | 0 | 6 | 2 |
| reference: no passage | 0 | 2 | 23 |

No passage is ever counted in the wrong direction, for either system.

**By camera and stratum** (nMAE; the release and `classical-v2`):

| Slice | Clips | Passages | Release | Classical |
|---|---:|---:|---:|---:|
| LeftFar stratum 1 | 18 | 31 | 0.097 | 0.194 |
| LeftFar stratum 2 (far range, tall frames) | 11 | 15 | 0.333 | 0.333 |
| LeftNear stratum 1 (dense) | 35 | 137 | 0.102 | 0.234 |

**Detection recall by fish size** (release, score ≥ 0.2, IoU ≥ 0.3, reference box area in m²):

| Size | Reference boxes | Recall |
|---|---:|---:|
| < 0.02 m² | 1,600 | 0.36 |
| 0.02–0.05 m² | 4,434 | 0.54 |
| 0.05–0.1 m² | 6,941 | 0.76 |
| ≥ 0.1 m² | 5,590 | 0.74 |

## Findings

1. **Tracking, not detection, causes most of the release's errors.** Merged (6), split (5)
   and duplicate (7) tracks make up 18 of the 30 passage-level errors. Pure detection
   failures (missed fish 5, background 1) make up 6.
2. **Duplicates are simultaneous.** In 6 of the 7 duplicate cases, the duplicate overlaps
   another track on the same fish in the same frames (for 2–15 frames). The detector puts
   two boxes on one fish, NMS at IoU 0.65 does not suppress the second, and the tracker
   follows both.
3. **Tracking errors are concentrated in the dense LeftNear clips**, where fish pass close
   together: all 6 merges, 4 of 5 splits and 6 of 7 duplicates.
4. **Small fish are often missed:** recall is 0.36 below 0.02 m² and 0.74–0.76 above
   0.05 m². The far-range stratum 2 (frames about 1,910 px tall, resized to 960 px for
   the detector) is the worst slice for both systems, although it has only 15 passages.
5. **Count-level nMAE understates the work.** The release's 30 passage-level errors become
   a count error of 22, because a missed fish and a duplicate in the same clip and direction
   cancel. A reviewer still has 30 cases to fix. Review prioritization (Stage 9) should
   target trajectories, not counts.
6. **The neural detector's gain over the classical one** is mostly fewer missed fish (19 → 5)
   and far less background (17 → 1).

## Experiments this justifies

Each changes one factor, is evaluated on kenai-val by counting nMAE with a paired clip
bootstrap and by passage-level errors, and is kept only with a measured benefit:

| # | Change | Targets | Needs |
|---|---|---|---|
| 1 | Stricter NMS (IoU 0.65 → lower) | duplicates (finding 2), some merges | Re-decoding only |
| 2 | Tracker gating and gap handling for neural detections | splits and merges (finding 3) | Re-tracking only |
| 3 | Higher detector input resolution | small fish, far range (finding 4) | Retraining (Kaggle) |
| 4 | Temporal input channels (Baseline++-style) | faint fish | Retraining (Kaggle) |

Detector size (Tiny → S) is not yet justified: even large fish reach only 0.74–0.76 recall,
so size alone is not the clear bottleneck. It runs only if experiments 3–4 leave detection
as the main error.

Every experiment selects on kenai-val, one day of 64 clips, which makes the val numbers
increasingly optimistic. The unbiased comparison is the frozen evaluation on the test
locations in Stage 11.
