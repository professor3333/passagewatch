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

## Experiment 1: extra duplicate suppression

Plan: `configs/tracking/tuning/suppression-plan-1.yaml`. A second greedy suppression pass
(`passagewatch.detection.suppression`) runs after the detector's NMS. It drops a box that a
higher-scoring box overlaps by IoU above a threshold, or by *containment* (intersection over
the smaller box's area) above a threshold. Detector, score threshold (0.2), tracker
(`classical-v2`) and counting are unchanged. Run with `scripts/compare_pipelines.py`.

The four settings were chosen after inspecting the kenai-val duplicates, so kenai-val results
carry selection bias:

| Variant | nMAE | Count errors | Passage errors | Missed | Merged | Split | Duplicate |
|---|---:|---:|---:|---:|---:|---:|---:|
| current (NMS 0.65 only) | 0.120 | 22 | 30 | 5 | 6 | 5 | 7 |
| + NMS IoU 0.5 | 0.120 | 22 | 28 | 5 | 7 | 5 | 3 |
| + NMS IoU 0.4 | 0.109 | 20 | 24 | 5 | 4 | 4 | 2 |
| **+ containment 0.8** (selected) | **0.104** | 19 | 25 | 4 | 5 | 5 | 3 |
| + containment 0.7 | 0.115 | 21 | 23 | 4 | 6 | 4 | 1 |

Selected minus current on kenai-val: **−0.016 [−0.042, +0.006]** (95% CI, paired clip
bootstrap), P(better) = 0.88. Every setting removes duplicates, but the nMAE gain is not
established on kenai-val. Its interval includes zero, and the selection was made on the same
clips.

**Confirmation on kenai-train (declared before it was run; see the plan file).** The detector
was trained on these 183 clips, so their detections are in-sample (recall 0.92). But they
were not used to choose the suppression setting. The rule: adopt the val-selected variant
only if its 95% CI against `current` on kenai-train lies entirely below zero.

| Variant | kenai-train nMAE | Passage errors | Duplicate | Difference vs current, train | Difference vs current, val |
|---|---:|---:|---:|---:|---:|
| current | 0.097 | 52 | 14 | | |
| + NMS IoU 0.5 | 0.071 | 34 | 1 | −0.027 [−0.049, −0.008] | +0.000 [−0.017, +0.014] |
| + NMS IoU 0.4 | 0.073 | 35 | 1 | −0.024 [−0.047, −0.006] | −0.011 [−0.034, +0.009] |
| + containment 0.8 | 0.086 | 45 | 9 | **−0.011 [−0.028, +0.004]** | −0.016 [−0.042, +0.006] |
| + containment 0.7 | 0.073 | 35 | 1 | −0.024 [−0.047, −0.006] | −0.006 [−0.027, +0.011] |

**Decision: the pipeline is unchanged.** The val-selected variant (containment 0.8) fails
the declared confirmation: its train interval includes zero. The suppression step stays
available and tested, but the release does not use it.

What the experiment did establish: every setting removes most duplicate tracks on both
partitions (val 7 → 1–3, train 14 → 1–9), and none makes nMAE worse by more than noise.
NMS IoU 0.4 and containment 0.7 improve on train and lean the right way on val. Adopting
either now would be a choice made after seeing both partitions. A new plan that declares
one of them in advance needs clips that neither training nor this selection has used: the
kenai-train days outside `kenai-dev-v1`.

## Experiment 2: the tracker's association gate

Plan: `configs/tracking/tuning/gate-plan-1.yaml`. Merged tracks on kenai-val jump 0.3–0.5 m
in one frame from one fish to another. Reference fish move a median 0.044 m per frame (99th
percentile 0.195 m). Only `gate_m` changes, from 0.5 m. The selection and confirmation rules
are the same as in experiment 1, and both were declared before either run.

| Variant | val nMAE | val passage errors (merged / split / duplicate) | train nMAE | train passage errors (merged / split / duplicate) |
|---|---:|---|---:|---|
| current (0.5 m) | 0.120 | 30 (6 / 5 / 7) | 0.097 | 52 (12 / 10 / 14) |
| **0.4 m** (selected on val) | 0.115 | 25 (3 / 7 / 2) | 0.097 | 46 (9 / 12 / 6) |
| 0.3 m | 0.120 | 26 (4 / 8 / 2) | 0.095 | 43 (4 / 15 / 3) |
| 0.25 m | 0.142 | 28 (3 / 10 / 1) | 0.113 | 51 (3 / 20 / 1) |
| 0.2 m | 0.159 | 31 (3 / 12 / 1) | 0.117 | 53 (2 / 23 / 1) |

The 0.4 m gate against `current`: val −0.006 [−0.025, +0.012], train **+0.000 [−0.017,
+0.016]**. **Decision: the pipeline is unchanged.**

A tighter gate turns merges into splits almost one for one. A new track starts with zero
velocity, so a tight gate loses a fast fish before the filter has learned its speed.
Passage-level errors fall at 0.3–0.4 m, which means less review work, but the counts do not
improve. Fixing this needs a tracker change rather than a different value, for example a
gate that scales with the filter's uncertainty, or bridging short gaps. Either would be a
new tracker version, tested as its own experiment.

## What the experiments say about the evaluation

kenai-val has 183 passages. Its paired-bootstrap intervals are about ±0.02 nMAE wide, so a
post-processing change worth less than about 2 points cannot be confirmed on it. Both
experiments so far changed counts by 0–3 errors. The remaining budget therefore goes to:

- changes with larger expected effects: detector retraining at a higher input resolution,
  and temporal channels, which target the 36% recall on small fish;
- a larger confirmation set that no training or selection has used: kenai-train days
  outside `kenai-dev-v1`.

| # | Experiment | Status |
|---|---|---|
| 1 | Extra duplicate suppression | Not adopted (confirmation failed); kept available |
| 2 | Tracker gate | Not adopted (no gain on either partition) |
| 3 | Higher detector input resolution | Not adopted: no confirmed gain on the holdout, 3.2× slower on CPU |
| 4 | Temporal input channels | **Adopted**: −0.093 [−0.133, −0.057] nMAE against v1 on the holdout |
| 5 | Uncertainty-scaled (Mahalanobis) association gate | Not adopted: fewer merges and duplicates, more splits, no count gain; kept available |

## Experiment 3 protocol (declared before any result)

Written on 2026-10-02 while `yolox-tiny-v2` (1280 × 640 input; everything else as
`yolox-tiny-v1`) trains, before any of its results exist:

1. **Selection, as for v1:** epochs 20, 25 and 30 × score thresholds 0.1–0.5, with the
   same tracker (`classical-v2`) and counting. The lowest counting nMAE on kenai-val wins.
2. **Confirmation on `kenai-holdout-v1`:** the selected v2 epoch and threshold, and v1's
   released setting (epoch 25, threshold 0.2), are each evaluated once on the holdout's 174
   clips (580 passages).
3. **Decision:** v2 replaces v1 only if the paired-clip-bootstrap 95% CI of
   nMAE(v2) − nMAE(v1) on the holdout lies entirely below zero. Otherwise v1 stays. The
   holdout and val numbers are both reported, as are passage-level errors on val and
   detection recall by fish size.
4. **Cost:** the larger input roughly doubles detection time per frame. CPU time per frame
   is measured and reported with the decision. It does not change the decision, because
   Stage 10 is where speed is optimized.

The same protocol then applies to experiment 4 (temporal channels), against whichever
detector this one keeps.

## Experiment 3 result: 1280 × 640 input (`yolox-tiny-v2`)

Trained on Kaggle (Tesla T4, about 36 images/s, 30 epochs) at commit `bc343b3`, which
declared the protocol above, with no uncommitted changes. Everything except the input size
equals `yolox-tiny-v1`.

**1. Selection on kenai-val** (the same grid and rule as v1):

| Epoch | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|
| 20 | 0.137 | **0.115** | 0.126 | 0.131 | 0.148 |
| 25 | 0.120 | 0.120 | 0.126 | 0.142 | 0.148 |
| 30 | 0.115 | 0.120 | 0.126 | 0.126 | 0.137 |

Epoch 20 at threshold 0.2 and epoch 30 at threshold 0.1 tie at 0.115. `evaluate_neural.py`'s
tie rule, fixed before v1 was selected (earlier epoch, then higher threshold), selects
**epoch 20, threshold 0.2**.

**2. Confirmation on `kenai-holdout-v1`** (174 clips, 580 passages; paired clip bootstrap,
10,000 resamples, 95% CI):

| | kenai-val | kenai-holdout-v1 |
|---|---|---|
| v1 (epoch 25, threshold 0.2) | 0.120 [0.078, 0.171] | 0.210 [0.166, 0.261] |
| v2 (epoch 20, threshold 0.2) | 0.115 | 0.193 [0.154, 0.238] |
| v2 − v1 | −0.006 [−0.040, +0.031] | **−0.017 [−0.054, +0.015]**, P(v2 better) = 0.83 |

**3. Decision: v1 stays.** The holdout interval includes zero, so under the declared rule
v2 does not replace v1. A real gain of a couple of nMAE points is possible, but it is not
established.

**4. Cost:** on the development Mac's CPU (Apple M1, 4 threads, batch 8, the same 40 frames
for both, measured back to back), v1 needs 82 ms per frame and v2 265 ms per frame:
**3.2× slower**. These are this benchmark's conditions, not the serving configuration,
which Stage 10 profiles.

**Why it does not help the counts** (kenai-val, threshold 0.2):

| | v1 | v2 |
|---|---:|---:|
| Detection recall, fish < 0.02 m² | 0.36 | 0.40 |
| Detection recall, 0.02–0.05 m² | 0.54 | 0.56 |
| Detection recall, 0.05–0.1 m² | 0.76 | 0.79 |
| Detection recall, ≥ 0.1 m² | 0.74 | 0.76 |
| Reference passages: missed outright | 5 | 2 |
| merged | 6 | 11 |
| split | 5 | 2 |
| ambiguous start/end | 2 | 5 |
| partial | 3 | 1 |
| Predicted passages: duplicate | 7 | 8 |
| background / non-passing fish | 1 / 1 | 1 / 1 |
| **Passage-level errors** | **30** | **31** |

The larger input finds a few more fish (recall up by 2–4 points in every size class), and
fewer fish are missed outright (5 → 2). In the dense near-range clips, though, the extra
detections are merged by the tracker (6 → 11 merges), so the counts barely change. With
this tracker, **detection resolution is no longer the bottleneck; association in dense
scenes is.** Far-range stratum 2, the motivating slice, improves (nMAE 0.333 → 0.267 on its
15 passages), but too few passages to confirm.

Consequences for the remaining experiments:

- Experiment 4 (temporal input) is built on v1's 960 × 416 input and compared with v1, the
  detector that was kept.
- A tracker change for dense scenes (experiment 2 showed that the gate alone only trades
  merges for splits) is the most promising direction after that.

## Experiment 5: an uncertainty-scaled association gate

Experiment 3 left association in dense scenes as the main error. Inside the released
detector's predicted tracks on kenai-val there are 135 identity switches (a track moving from
one reference fish to another). 76% happen in established tracks (5+ observations, median
14), with a median jump of 0.31 m. Reference fish move a median 0.044 m per frame. A fixed
0.5 m gate lets a confident track take a neighbor; experiment 2 showed that tightening it for
every track breaks young tracks instead.

The tracker therefore gained an optional Mahalanobis gate (`TrackerConfig.gate_sigma`): a
detection must also lie within `gate_sigma` standard deviations of the track's prediction,
under the filter's innovation covariance. An established track's innovation SD settles near
0.12 m per axis with `classical-v2`'s noise settings, so its gate tightens to about
0.18–0.36 m. A new track, whose velocity is still uncertain, keeps the 0.5 m distance gate.
It is off by default, and tests cover both cases.

Plan `configs/tracking/tuning/sigma-gate-plan-1.yaml`, with the rules declared before any
run. On kenai-val:

| Variant | nMAE | Count errors | Passage errors | Merged | Split | Duplicate | Missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| **current** (selected) | 0.120 | 22 | 30 | 6 | 5 | 7 | 5 |
| gate_sigma 1.5 | 0.131 | 24 | 26 | 2 | 8 | 3 | 6 |
| gate_sigma 2.0 | 0.126 | 23 | 25 | 2 | 7 | 3 | 5 |
| gate_sigma 2.5 | 0.126 | 23 | 27 | 2 | 8 | 4 | 5 |
| gate_sigma 3.0 | 0.126 | 23 | 27 | 2 | 8 | 4 | 5 |

**Decision: the pipeline is unchanged.** No variant improves the count on kenai-val, so
nothing goes to the holdout.

The gate does what it was built for: merges fall from 6 to 2 and duplicates from 7 to 3,
and passage-level errors, the reviewer's work, from 30 to 25. But a track whose fish is
briefly missed, or moves abruptly, now ends instead of jumping, and the fish restarts as a
new track, so splits rise and the count does not improve. Fixing that needs a second step:
joining a track that ends to one that starts shortly after near its predicted position, which
batch analysis allows. That is a new tracker design, and with experiments 1–5 the planned
budget of about six substantive experiments is nearly used. It is recorded here as the next
hypothesis rather than run now.

## Experiment 4 result: temporal input channels (`yolox-tiny-t1`)

Trained on Kaggle (Tesla T4, about 44 images/s, 30 epochs) at commit `8341d37`, with no
uncommitted changes. Everything except the input encoding (`letterbox-temporal3-v1`: the
frame, the frame minus the clip's background, and the motion to the next frame; after the CFC
authors' Baseline++) equals `yolox-tiny-v1`. Same protocol as experiment 3, against v1.

**1. Selection on kenai-val:**

| Epoch | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|
| 20 | 0.098 | 0.093 | 0.082 | 0.082 | 0.082 |
| 25 | 0.077 | 0.071 | 0.066 | **0.066** | 0.071 |
| 30 | 0.082 | 0.077 | 0.066 | 0.066 | 0.066 |

The tie rule (earlier epoch, then higher threshold) selects **epoch 25, threshold 0.4**.

**2. Confirmation on `kenai-holdout-v1`** (174 clips, 580 passages; paired clip bootstrap,
10,000 resamples, 95% CI):

| | kenai-val | kenai-holdout-v1 |
|---|---|---|
| v1 (epoch 25, threshold 0.2) | 0.120 [0.078, 0.171] | 0.210 [0.166, 0.261] |
| t1 (epoch 25, threshold 0.4) | 0.066 [0.027, 0.111] | **0.117 [0.089, 0.149]** |
| t1 − v1 | −0.055 [−0.101, −0.015] | **−0.093 [−0.133, −0.057]**, P(t1 better) = 1.000 |

On the holdout, t1 misses 44 passages (v1: 99) and adds 24 false ones (v1: 23), with
detection recall 0.79 and precision 0.88 (v1: 0.71 and 0.80).

**3. Decision: t1 replaces v1.** The holdout interval lies entirely below zero: a 44% lower
counting error on unseen days. Against `classical-v2` on the holdout (0.369), the
improvement is about 0.25 nMAE.

**4. Cost:** on the development M1's CPU (4 threads, batch 8, the 541-frame profile clip),
98.5 ms per frame against 92.5 for v1 (+6%). Detection costs the same; the extra time is
temporal encoding (a second decode pass and blurring: 8.3 ms/frame against 1.8). Memory is
unchanged: one background frame per recording.

**Where the gain comes from** (kenai-val, selected settings):

| | v1 | t1 |
|---|---:|---:|
| Reference passages: missed outright | 5 | 2 |
| merged / split | 6 / 5 | 5 / 2 |
| ambiguous / partial | 2 / 3 | 1 / 0 |
| Predicted passages: duplicate | 7 | 2 |
| background / non-passing fish | 1 / 1 | 1 / 1 |
| **Passage-level errors** | **30** | **13** |
| Detection recall, fish < 0.02 m² | 0.36 | 0.40 |
| Detection recall, ≥ 0.1 m² | 0.74 | 0.86 |
| nMAE, far-range stratum 2 (15 passages) | 0.333 | 0.000 |

The motion and background channels make fish stand out from static clutter and from each
other. Fewer fish are missed, tracks break less often, and the part-of-fish duplicate boxes
that experiment 1 tried to suppress largely disappear (7 → 2). The smallest fish remain hard
(recall 0.40 below 0.02 m²).

These are dev-data results: kenai-val selected the setting, and kenai-holdout-v1 confirmed
it, both from the same two cameras and season. The official test locations, opened once in
Stage 11, measure transfer to other cameras and rivers.

## Stage 8 summary

Five experiments after the tracker comparison of Stage 5, each changing one factor with its
rule declared before its results:

| # | Change | Result |
|---|---|---|
| 1 | Extra duplicate suppression | Not adopted |
| 2 | Tracker gate | Not adopted |
| 3 | 1280 × 640 input | Not adopted (3.2× slower, gain not confirmed) |
| 4 | Temporal input channels | **Adopted** (−0.093 nMAE on the holdout) |
| 5 | Uncertainty-scaled gate | Not adopted (kept available) |

The next release uses `yolox-tiny-t1` (epoch 25, threshold 0.4) with the unchanged
`classical-v2` tracker.
