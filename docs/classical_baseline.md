# Classical Baseline

The classical baseline is the first of the three systems PassageWatch compares (classical,
neural single-frame, neural temporal). It uses no learned weights. It exists so that the
neural systems have to beat a **genuinely tuned** conventional pipeline on the same data,
by the same counting metric, rather than a strawman.

Code: `passagewatch.detection.classical`, `passagewatch.tracking.kalman`,
`passagewatch.inference.classical`. Configurations: `configs/tracking/classical-*.yaml`.

## Method

**Detection**, per frame:

1. Frames are read in grayscale and downscaled so that their height is at most
   `max_analysis_height`. Boxes are mapped back to original pixels at the end.
2. **Background:** the per-pixel median of every `background_stride`-th frame in a centered
   window of `background_window` frames, recomputed every `background_update_every`
   frames. A centered window looks ahead in time. That is acceptable because recordings
   are analyzed after upload, not live.
3. **Foreground:** `frame − background`, smoothed with a Gaussian of `blur_sigma_px`.
   Pixels inside the sonar fan count as foreground where the difference exceeds
   `threshold_sigma` times a robust noise level (1.4826 × the median absolute deviation),
   and is at least `min_contrast`. The noise level is estimated per horizontal band of
   rows when `noise_bands > 1`, because sonar noise changes with range.
4. **Cleanup:** morphological opening (`open_px`) removes speckle. Closing (`close_px`)
   joins the parts of one fish.
5. **Components:** connected regions whose area is within `[min_area_m2, max_area_m2]`
   become detections. The score is the region's mean contrast in noise units.

**Tracking:** constant-velocity Kalman filters on box centers in meters, with Hungarian
assignment gated at `gate_m`. A track ends after more than `max_age` consecutive missed
frames, or at the end of the recording. Tracks shorter than `min_length` (or with fewer
than `min_hits` matches) are dropped. A trajectory contains only the detections actually
matched to it, never predicted positions.

**Counting:** completed trajectories are counted with `cfc-compatible-v1`
([counting policy](counting_policy.md)).

**Sizes and distances are in meters.** Each clip's metadata gives its extent in meters,
so the area filter and the tracking gate mean the same thing at every render resolution.
In Kenai, every clip is rendered at about 0.007 m per pixel. Annotated fish boxes have a
median size of 0.5 m × 0.15 m, and the smallest are about 0.08 m × 0.07 m.

## Running it

```bash
# One command: clips -> tracks (MOT files) -> directional counts -> report.json
uv run python scripts/run_classical.py --manifest tiny-v1 --partitions val
uv run python scripts/run_classical.py --manifest full-v2 --partitions val \
    --frames-dir data/extracted/cfc/kenai-dev-v1 --config configs/tracking/classical-v2.yaml
```

The report gives per-location counting nMAE, detection recall and precision against the
reference boxes (IoU ≥ 0.3, one-to-one per frame; a diagnostic, not the release metric),
per-clip counts, the configuration and its SHA-256, and CPU time per stage. The test
partition cannot be selected.

## Tuning protocol

- **Data:** the kenai-train part of the `kenai-dev-v1` subset (whole recording days; see
  the [dataset card](dataset_card.md)). kenai-val is not used during the search. It is used
  once at the end, to compare the untuned and the tuned configuration. Test locations are
  never used.
- **Objective:** counting nMAE (the release metric), not detection quality.
- **Search:** a declared plan (`configs/tracking/tuning/classical-plan-1.yaml`) of six
  experiments, each varying one parameter, or one coupled pair, of the current best
  configuration over a few values. The best value is kept before the next experiment.
  Every run is recorded, and a winner at the edge of its grid is flagged, because the
  optimum may lie outside it.
- **Result:** a new configuration name (`classical-v2`). A name is never reused for
  different values.

## Results

Measured on 2026-10-01 with the `full-v2` manifest (frames of the `kenai-dev-v1`
subset). All runs use full-length clips and counting policy `cfc-compatible-v1` with the
line at `x = 0.5`.

### Tuning on kenai-train (183 clips, 452 passages)

| Step | Change kept | Train nMAE |
|---|---|---:|
| start | `classical-v1` (untuned) | 10.956 |
| E1 threshold | `threshold_sigma` 3 → 5 (tried 3, 4, 5, 6, 7) | 0.469 |
| E2 noise by range | `noise_bands` 1 → 4 (tried 1, 4, 8; 4 and 8 tie) | 0.438 |
| E3 minimum area | none (tried 0.004–0.025 m²; larger minimums lose small fish) | 0.438 |
| E4 smoothing | none (tried blur 0, 1, 2 px) | 0.438 |
| E5 association | `max_age` 3 → 4, gate stays 0.5 m (tried gate 0.3/0.5/0.8 × max_age 2/4) | 0.409 |
| E6 track length | `min_length` 3 → 8 (tried 3, 5, 8, 12) | **0.350** |

21 distinct configurations were run (about 2 h 50 min on 6 CPU workers). The untuned
configuration predicted 5,368 passages against 452 true ones, almost all of them false:
background clutter at a 3σ threshold. `classical-v2` predicts 360. The full log is
produced by `scripts/tune_classical.py` (`runs/tuning/…/tuning.json`, not committed).

Caveats: the best `min_area_m2` is the smallest value tried, but E3 changed nothing, so the
grid-edge warning (which applies to improvements) did not fire. `max_age = 4` is the top of
a two-value grid. Both directions are unexplored.

### One-time comparison on kenai-val (64 clips, 183 passages)

kenai-val was not used during tuning. It was evaluated once for each configuration:

| Configuration | Errors | Predicted passages | nMAE | Detection recall / precision |
|---|---:|---:|---:|---|
| `classical-v1` (untuned) | 479 | 640 | 2.618 | 0.53 / 0.20 |
| `classical-v2` (tuned on train) | 43 | 166 | **0.235** | 0.23 / 0.40 |

For reference, on the same 64 kenai-val clips with the same evaluator, CFC's published
**Baseline** (a learned YOLOv5 detector with a tracker) scores **0.049**, and
**Baseline++** scores **0.033** ([counting policy](counting_policy.md#validation-against-the-official-evaluator)).
The tuned classical pipeline is about five times worse than the learned baselines. That is
the bar the neural detector in Stage 4 has to clear.

Where `classical-v2`'s 43 errors on kenai-val come from:

- **30 missed passages and 13 false passages**, spread over 32 of the 64 clips. The
  largest error in a single clip is 3.
- **Detection recall is low (0.23 at IoU ≥ 0.3).** Many fish are found in only some of
  their frames. Counting still works much of the time, because only a trajectory's first
  and last positions matter, but faint fish and fish whose track breaks near the line are
  lost. Tuning moved the system from heavily over-counting to mildly under-counting.

**Runtime:** about 45 ms of CPU per frame (decode, detect and track) on kenai-val, measured
while 6 worker processes shared an 8-core Apple Silicon laptop. This is a development
measurement, not a serving benchmark.

### Unbiased check on `kenai-holdout-v1` (Stage 8)

Evaluated once, with the same settings, on five unseen kenai-train days (174 clips, 580
passages): **nMAE 0.369 [0.316, 0.430]**, detection recall 0.30, precision 0.35, and 37 ms
per frame on the development Mac's CPU. This is worse than on kenai-val (0.235), although
kenai-val was never used to choose these settings: the holdout days are harder. The
comparison with the neural system is in `docs/neural_baseline.md`.

### Limitations of this result

- Tuned on 183 of the 482 kenai-train clips (5 of 15 days). Full-train tuning could differ.
- kenai-val is a single day (2018-06-03) inside the training period, so it measures new
  recordings, not a new season. No test location was used.
- The selection used kenai-train only. Comparing two configurations on kenai-val, as above,
  does not select anything, but any further tuning that looks at these val numbers would
  make kenai-val optimistic.
