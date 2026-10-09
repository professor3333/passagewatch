# Usability Study (Stage 12): Results

The study compared **manual counting** with **assisted review** in the PassageWatch review
interface, as declared in [usability_study.md](usability_study.md): release
`passagewatch-0.3.0`, 12 `kenai-holdout-v1` clips in two matched sets, a 2 × 2 Latin
square, the developer (`D1`) and four independent participants (`P1`–`P4`) who did not
build the system. Sessions ran on 2026-10-05, and P4's second block on 2026-10-09.

**Result: the target was not met.** For the independent participants, assisted review was
**not faster** than manual counting, and their **final counts were less accurate**. The
release's automatic counts on the same clips were *more* accurate than either condition's
final counts. Most of the assisted errors came from one interface action: participants set
a direction on tracks that the release had not counted as passages, and each such action
added a passage.

These results describe five people working on short, preselected clips. They are not
evidence about technicians or a working day's footage (see **Limitations**).

## Primary results

Time saving = 1 − (mean assisted active time per clip ÷ mean manual active time per clip).
nMAE = Σ(|R̂ − R| + |L̂ − L|) / Σ(R + L) against the CFC reference. Intervals are 95%
bootstrap intervals over clips. The group interval is over participants, then their clips.

| Participant | Clips (manual / assisted) | Mean active s per clip (manual / assisted) | Time saving [95% CI] | nMAE manual | nMAE assisted | Difference [95% CI] | Target met |
|---|---|---|---|---|---|---|---|
| D1 (developer) | 6 / 6 | 33.6 / 40.6 | −21% [−64%, +12%] | 0.091 | 0.348 | +0.257 [−0.039, +1.000] | no |
| P1 | 6 / 6 | 26.0 / 34.7 | −33% [−107%, +22%] | 0.261 | 0.136 | −0.125 [−0.791, +0.118] | no |
| P2 | 6 / 6 | 27.2 / 18.2 | +33% [+4%, +60%] | 0.217 | 0.182 | −0.036 [−0.526, +0.204] | yes |
| P3 | 6 / 6 | 26.7 / 27.0 | −1% [−93%, +47%] | 0.091 | 0.826 | +0.735 [+0.316, +1.650] | no |
| P4 | 5 / 6 | 16.1 / 30.5 | −90% [−196%, −11%] | 0.143 | 0.652 | +0.509 [+0.179, +1.144] | no |

**Independent group (P1–P4):** time saving **−14% [−77%, +29%]** (geometric mean of the
time ratios, 1.14). nMAE **manual 0.180** (16 errors / 89 passages) vs **assisted 0.456**
(41 / 90), difference **+0.276 [−0.123, +0.704]**. **Target met: no.** The point estimates
show assisted review as slower and less accurate. The intervals are wide, and neither one
rules out a small effect in the other direction.

The developer's results (D1) are reported alone and are not part of the group figures. They
also did not meet the target.

## Secondary results

| Participant | SUS manual | SUS assisted | TLX raw manual | TLX raw assisted | Accept | Reject | Set direction (on non-passages) | Add passage | Errors fixed | Errors introduced |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 97.5 | 100.0 | 15.8 | 18.3 | 10 | 0 | 0 (0) | 5 | 0 | 5 |
| P1 | 90.0 | 90.0 | 15.8 | 15.8 | 27 | 0 | 4 (3) | 1 | 0 | 0 |
| P2 | 87.5 | 87.5 | 20.8 | 15.0 | 24 | 0 | 5 (5) | 0 | 1 | 2 |
| P3 | 85.0 | 90.0 | 3.3 | 17.5 | 30 | 0 | 16 (16) | 0 | 0 | 16 |
| P4 | 92.5 | 85.0 | 5.0 | 3.3 | 32 | 0 | 13 (12) | 0 | 0 | 12 |

Review actions are from scored assisted clips, up to the revision each participant's counts
came from. "Errors fixed / introduced" is per direction: how much closer to (or further from)
the reference the final counts are than the release's automatic counts.

- **Automatic counts were better than reviewed counts.** On the independent participants'
  assisted clips, the release's automatic counts have nMAE **0.133** (12 errors / 90
  passages). Review fixed 1 error and introduced 30.
- **Set direction on tracks that were not passages.** 36 of the 38 `set_direction` actions
  on scored clips were on tracks the release had not counted: 29 tracks that never crossed
  the line and 7 stationary ones. Each one added a passage. For P3 and P4 this explains
  every introduced error (16 and 12). The review panel offers **Set → right (r)** and
  **Set ← left (l)** on every track, so the likely reading is "this fish swims right", not
  "count this track as a passage". The study recorded no comments that could confirm this.
- **The developer** introduced their 5 errors with **Add passage**: four on one clip, all
  at its last frame.
- **Accepting.** The independent participants accepted 4.7 tracks per assisted clip on
  average, although accepting does not change a count.
- **SUS** is high in both conditions (85–100). **NASA-TLX raw** is low, but see the
  performance-scale note under **Deviations**.

### Exploratory, post hoc

These figures were computed after seeing the data. They suggest what to change next, and
they are not results of the declared analysis.

- Removing the 36 `set_direction` actions on non-passages would leave the independent
  group's assisted nMAE at **0.122** (11 / 90), below its manual nMAE of 0.180.
- Manual counting took a median **1.08×** the clip's own duration (independent
  participants, scored clips), and assisted review took 1.11×. The study's clips last
  20–34 s and have 1–8 passages. On clips like these, there is little time for assistance
  to save.
- NASA-TLX without the performance scale (mean of the other five) is 0–8 for every
  participant and condition.

## Deviations from the declared design

- **Exclusion of P4 · c01.** The session was interrupted and Pause was not pressed, so 66
  minutes were recorded for a 25 s clip. The design allows excluding a clip only for a
  technical fault. This exclusion was decided by the facilitator after the session, and is
  passed to the analysis with `--exclude`.
- **Time ratio from per-clip means.** The design defines the ratio from total times over
  6 + 6 clips. With one clip excluded, P4 has 5 manual clips, which biases a ratio of
  totals, so the analysis now uses mean time per clip. This is identical for everyone else.
  P4's time saving is **−128%** from totals and **−90%** from means, and the group's is
  **−20%** and **−14%** respectively. The conclusion is the same either way. This was
  changed after the data was seen.
- **Session length and breaks.** Sessions took about 7–12 minutes of work, not the planned
  75, because the clips are short and were counted at about their own playback speed. No
  5-minute break was taken: D1, P1 and P3 started their second block within seconds of the
  first block's questionnaire. P2's two blocks were about 8 hours apart, and P4's 4 days
  apart.
- **Closing questions were not collected.** No participant was asked which condition they
  preferred or for comments, and no facilitator notes were kept. Nothing beyond the recorded
  actions explains the behaviour above.
- **Review actions by type** are a declared measure. Exporting them
  (`scripts/export_study_reviews.py`) was added after the sessions.
- **NASA-TLX performance scale.** The page labels it as in the standard (0 = perfect,
  100 = failure). Most answers were 65–100 on performance and near 0 on every other scale,
  which suggests participants read high as "successful". The TLX raw scores above are as
  recorded, and probably overstate workload.

## What this means for PassageWatch

- The assisted condition did not save time on these clips, and as shipped it made
  reviewers' counts worse than the automatic ones. The README and model card make **no
  time-saved claim**.
- **The interface change suggested by this study is not made here.** The suggestion: do
  not offer "set direction" on tracks that are not passages, or label it "Count as passage
  → / ←", so that a track is counted only on purpose. A bulk "accept the suggested tracks"
  action would also cut per-track clicks. Any re-test needs a newly declared design and
  participants who have not seen these clips.

## Limitations

- **Five people, 12 clips each**, four of them independent. These sessions describe these
  people, not reviewers in general. No fisheries technician took part.
- **Short, preselected clips** from one river, two cameras and one season (`kenai-holdout-v1`).
  Clips with 1–8 fish lasting 15–45 s favour manual counting. Longer or denser footage was
  not studied.
- **Reference quality.** CFC's annotations can be wrong, and the reference is used anyway.
- **The deviations above**, in particular the lack of participant comments.

## Reproducing

The records are committed in `study/`: `plan.json` (the selection, written before the first
session), `trials.csv`, `questionnaires.csv`, `review_actions.csv` and `results.json`.

```bash
# From the study service (scripts/study_service.sh start), after the last session:
curl -s http://127.0.0.1:8010/v1/study/export/trials.csv > study/trials.csv
curl -s http://127.0.0.1:8010/v1/study/export/questionnaires.csv > study/questionnaires.csv
uv run python scripts/export_study_reviews.py --url http://127.0.0.1:8010

# Offline, from the committed files:
uv run python scripts/analyze_usability_study.py \
  --exclude 'P4:c01:session interrupted and Pause not pressed (66 min recorded for a 25 s video)' \
  --out study/results.json
```
