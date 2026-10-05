# Usability Study (Stage 12): Design

Does PassageWatch's assisted review let a person produce directional fish counts faster than
counting the video by hand, without making the counts worse? This page fixes the study's
design, materials, measures and analysis **before any data is collected**, so the results
cannot shape the method. Results will go in a separate report; this design is not changed
after the first session.

If only the developer takes part, the study is reported as a **developer usability study**,
never as validated fisheries impact.

## Question and declared targets

| | |
|---|---|
| Comparison | **Manual counting** (video, counting line, a tally) against **assisted review** (the release's automatic counts, review queue and corrections in the PassageWatch review interface) |
| Primary measure 1 | Active time per clip, in seconds |
| Primary measure 2 | Final count error per clip against the CFC reference: \|R̂ − R\| + \|L̂ − L\|, and nMAE per condition |
| Planning target (`docs/design.md`) | Assisted review needs **at least 30% less active time** than manual counting **without worse final counts** |

The target is a planning assumption, not an expected result. It is reported as met or not
met by the rules under **Analysis**, whatever the numbers turn out to be.

## Participants

- **The developer** (participant `D1`), who knows the system. Their results are always
  labelled "developer".
- **Optionally, other participants** (`P1`, `P2`, …) who did not build the system: for
  example classmates or colleagues, with no fisheries experience required. Each gets the
  written instructions below and nothing more. With fewer than 3 such participants, results
  are reported per person, as a pilot, without group-level claims.
- Pseudonymous codes only: no names, contact details or personal data are recorded or
  committed.

## Materials

**Clips** come from `kenai-holdout-v1`. Those recordings were not used to train the
detector or to choose any setting, the frames are on the development machine, and the
release's automatic counts on them are realistic. Neither kenai-val, where settings were
selected, nor the test locations, whose frames are not kept, are used.

Selection, by `scripts/prepare_usability_study.py` (seed 0, run once; implemented in
`passagewatch.evaluation.study_design`):

1. **Eligible clips:** 150–450 frames lasting 15–45 s, with 1–8 reference passages, and not
   named anywhere in the documentation.
2. **Strata:** each eligible clip falls into a stratum by camera (LeftFar or LeftNear) and
   reference passages (1–2, 3–4, 5–8).
3. **Trial clips:** two per stratum, 12 in all.
4. **Two matched sets:** each of **S1** and **S2** gets one clip of every stratum, so camera
   and passage mix match by construction. Of the 64 possible splits, only those keeping total
   reference passages (within 2), total duration (within 10%) and total automatic count error
   (within 2) equal are allowed, and the most balanced is chosen.
5. **Practice clips:** one per camera, from the remaining eligible clips.

*Amendments on 2026-10-05, before any session.* The selection was first written as "draw 14
clips, set the first two aside for practice, split the other 12". A dry run showed that this
leaves an odd number of clips per camera, so no split can match the camera mix. Steps 3–5
replace it. The clip-length rule was also "150–450 frames (about 15–50 s)", but frame rates
vary, and a 200-frame clip can last 60 s, which made duration matching impossible. A 15–45 s
limit was added. With these rules, 59 clips are eligible. The two sets have 22 and 23
passages, 144 s and 139 s, and 3 automatic count errors each.

The study uses image directions (**→ right**, **← left**), as the benchmark does. The CFC
metadata does not record which way is upstream, so the introduction does not claim it.

The selected clips, their sets and the release's automatic counts are written to
`study/plan.json` and committed before the first session. The reference counts stay out of
the participant's view.

## Design

Within-subject, two conditions, counterbalanced with a 2 × 2 Latin square over **order** and
**clip set**. Participants are assigned in this sequence:

| Participant | First block | Second block |
|---|---|---|
| D1 | Manual · S1 | Assisted · S2 |
| P1 | Assisted · S1 | Manual · S2 |
| P2 | Manual · S2 | Assisted · S1 |
| P3 | Assisted · S2 | Manual · S1 |
| P4… | repeats from the D1 row | |

- Nobody sees a clip twice, so nobody recounts a clip they have already counted.
- Each block starts with one practice clip in that condition (the two practice clips are the
  same for everyone and are not scored).
- Clips within a block are shown in a fixed shuffled order.
- A short break is taken between blocks.

## Procedure

Total time per participant: about **75 minutes**.

1. **Introduction** (5 min). What a passage is (a fish that crosses the vertical line and
   ends on the other side; one that crosses and comes back does not count); that counts are
   kept separately for fish moving right and fish moving left; and that the task is to get
   each clip's two counts right without hurrying.
2. **Block 1** (about 25 min): the practice clip, then 6 scored clips, in that block's
   condition.
3. **Questionnaires** (3 min): the System Usability Scale (SUS, 10 items) and the six
   NASA-TLX raw workload scales for that condition.
4. **Break** (5 min).
5. **Block 2** (about 25 min) and its questionnaires.
6. **Closing** (5 min): which condition the participant preferred and why, and free comments.

**Manual condition.** A study page shows the recording with the counting line, playback
controls (play, pause, frame steps, speeds 0.5–4×, scrubbing) and two counters, → right and
← left, each with +1 and −1. No model output is shown. The participant presses **Done**
when satisfied.

**Assisted condition.** The participant opens the job in the shipped review interface. It
shows the release's tracks, triage states, review queue and audit windows, and the actions
accept, reject, set direction, mark unresolved and add missed fish. The participant reviews
as they see fit and presses **Done**, which exports the reviewed counts. The release's
results are computed before the session, so no waiting is involved.

**Timing.** Active time runs from the moment a clip's page opens until **Done**, and is
recorded by the study pages, not a stopwatch. A **Pause** button stops it for interruptions.
There is no time limit, but the facilitator notes any clip that took over 10 minutes.

## Measures

| Measure | Per | Source |
|---|---|---|
| Active time (s) | clip | study page timestamps |
| Final counts (R̂, L̂) | clip | manual counters, or the assisted export's reviewed counts |
| Count error \|R̂ − R\| + \|L̂ − L\| | clip | against CFC reference counts |
| Automatic count error | clip (assisted) | the release's counts before review |
| Review actions by type | clip (assisted) | the job's review events |
| Errors fixed and introduced | clip (assisted) | final against automatic error |
| SUS score (0–100) | condition | questionnaire |
| NASA-TLX raw (0–100) | condition | questionnaire |
| Preference and comments | participant | closing questions |

## Analysis

These rules are fixed now, by `scripts/analyze_usability_study.py`:

- **Time.** For each participant, the ratio of total assisted time to total manual time over
  their 6 + 6 clips. The **time saving** is 1 − that ratio, reported for each participant
  with a bootstrap interval over their clips. With 3 or more **independent** participants
  (`P` codes), the geometric mean of their ratios is also reported, with a bootstrap
  interval over participants. **The developer's results never enter a group figure.**
- **Counts.** nMAE per condition over each participant's 6 clips, and the paired difference
  (assisted − manual) with a bootstrap interval over clips (and over participants for the
  independent group).
- **Target met** only if the time saving is at least 30% **and** assisted nMAE is not higher
  than manual nMAE (point estimate). Both intervals are reported regardless.
- **Secondary results** are reported descriptively: SUS, NASA-TLX, actions, errors fixed and
  introduced, and comments. No significance tests are run on a sample this small.
- *Clarified on 2026-10-05, before any session:* the analysis said "3 or more
  participants", which could include the developer. As under **Participants**, group
  figures use independent participants only.
- **Exclusions:** practice clips, and a clip only if its session was interrupted for a
  technical fault (recorded with the reason). No clip is dropped for being slow or wrong.

## Threats to validity

- **The developer knows the system and the data.** Their results are labelled and kept
  apart; they likely flatter the assisted condition.
- **Small sample.** One to five people, 12 clips each: the results describe these sessions,
  not reviewers in general.
- **Reference quality.** CFC's annotations can be wrong. Where a participant disagrees with
  the reference, the clip is noted, but the reference is still used.
- **One river.** All clips come from the Kenai, two cameras and one season. The test results
  (`docs/test_results.md`) show the model does worse on other rivers, so assisted review
  would need more corrections there.
- **Learning and fatigue** are balanced by the Latin square and practice clips, not removed.
- **Not real work.** These are short, preselected clips, not a technician's working day, and
  no fisheries professional takes part unless one volunteers.

## Study tooling

Built and tested before the first session:

- **`scripts/prepare_usability_study.py`.** Selects and matches the clips by the rules above
  (`--dry-run` prints the selection only). It uploads the clips to the study service and
  analyses them with the release. It checks that the service's automatic counts equal the
  evaluated ones, and writes `study/plan.json`. Every participant gets their own job for each
  clip they review, so nobody sees another participant's corrections.
- **The study service:** the release with `PASSAGEWATCH_STUDY_MODE` on
  (`docker-compose.study.yml`, its own Compose project, 90-day upload retention). Records
  go to their own database, `study.db`, never to the product's tables:
  - `trials`: participant code, block, condition, clip, start and finish times, active
    milliseconds, pauses, final counts, and for assisted trials the job and review revision
    the counts came from;
  - `questionnaires`: SUS and NASA-TLX raw answers per participant and condition.

  Both tables are exported as CSV (`/v1/study/export/{trials,questionnaires}.csv`). The plan
  shown to the pages leaves out the reference and automatic counts.
- **Study pages** at `/#/study`:
  - The facilitator picks the participant code. The pages then follow the Latin square:
    block instructions, the practice clip, the six clips, the questionnaires, then the
    second block.
  - The manual page has the player, counting line and the two tallies.
  - The assisted page is the shipped review interface with a study bar; its export links
    are hidden, and **Done** records the reviewed counts.
  - **Pause** stops the clock and covers the page until **Resume**.
  - Reloading the page restarts the current clip; the facilitator notes this as a
    technical fault.
- **`scripts/analyze_usability_study.py`:** the analysis above, from the plan and the two
  CSV files (implemented in `passagewatch.evaluation.usability`). A clip excluded for a
  technical fault is given with `--exclude P1:c04:"reason"` and listed in the output.

### Running the study

The study runs on the development machine, without Docker
(`scripts/study_service.sh`: the API with the study pages and one worker, data in
`var/study/`, uploads kept for 90 days). It was prepared once:

```bash
scripts/study_service.sh start
uv run python scripts/prepare_usability_study.py --url http://127.0.0.1:8010
```

`docker-compose.study.yml` runs the same service in containers instead.

Before each session, `scripts/study_service.sh start`; after it, `scripts/study_service.sh
stop` (all data is kept). For each session, open `http://127.0.0.1:8010/#/study` in a fresh browser window, choose
the participant's code and hand over. Read the introduction aloud from the first block's
instructions. Note any clip that took over 10 minutes and any technical fault. After the
last session:

```bash
scripts/study_service.sh start
curl -s http://127.0.0.1:8010/v1/study/export/trials.csv > study/trials.csv
curl -s http://127.0.0.1:8010/v1/study/export/questionnaires.csv > study/questionnaires.csv
uv run python scripts/analyze_usability_study.py --out study/results.json
```

The roadmap also asks for three demo examples: a clear clip, a difficult clip and an
unfamiliar camera. They are prepared as visibly labelled cached results. A person other
than the developer checks the README, model card, dataset card and architecture guide.
