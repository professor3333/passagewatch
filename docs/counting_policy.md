# Counting Policy

The counting policy turns **completed trajectories** into **directional passage counts**.
It is versioned. A policy version, once released, is never changed. Any change in behavior
gets a new version string, and every job, result, and export records the version it used.

Current policy: **`cfc-compatible-v1`**.

## 1. Internal coordinate convention

All PassageWatch code works in one internal convention. Source annotations are converted
into it **once**, at ingestion.

| Quantity | Internal convention |
|---|---|
| Frame index | **0-based** integer, in recording order |
| Timestamp | Seconds from the start of the recording, as a float |
| Pixel coordinates | **0-based continuous** coordinates in the *original* frame. Pixel column `i` covers `[i, i+1)`; the image spans `[0, W] × [0, H]` |
| Bounding box | `(x_min, y_min, x_max, y_max)` in absolute pixels, as floats, with `x_max > x_min` and `y_max > y_min` |
| Box center | `cx = (x_min + x_max) / 2`, `cy = (y_min + y_max) / 2` |
| Normalized position | `u = cx / W`, `v = cy / H`, both in `[0, 1]` |

### Conversion from CFC / MOTChallenge annotations

CFC `gt.txt` rows are `frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z`.
**Frames, IDs, and box coordinates are 1-indexed** (the smallest `bb_left` and `bb_top` are 1).
Image files, however, are named from `0.jpg`: **gt frame `N` is image `N−1.jpg`**. This was
verified against the full annotations: across all 1,567 clips, gt frames lie in
`[1, num_frames]`, and 470 clips have an annotation on frame `num_frames` itself, which
could not happen with 0-based numbering.

```
frame_index = frame - 1
x_min = bb_left - 1          x_max = x_min + bb_width
y_min = bb_top  - 1          y_max = y_min + bb_height
track_id = id                (kept as given; it is an identifier, not an index)
```

This matches the official evaluator's normalization, `((bb_left − 1)/W, (bb_top − 1)/H,
w/W, h/H)`. Exporting back to MOT format (for the official evaluator) applies the exact
inverse: `bb_left = x_min + 1`, `bb_top = y_min + 1`. A unit test checks the round trip.

The conversion lives only in `passagewatch.ingestion.mot`. Boxes that extend beyond the
image are kept **unclipped**, as in the official evaluator. `W` and `H` for normalization
are the clip **metadata** size, which is what the official evaluator uses. A few tiny-subset
images differ from it by one pixel; see `docs/dataset_card.md`.

## 2. The rule (`cfc-compatible-v1`)

This rule reproduces the counting function in the official CFC evaluator
(`CFC/evaluate.py`, `nMAE.count`, in
[visipedia/caltech-fish-counting](https://github.com/visipedia/caltech-fish-counting)).

For each **completed** trajectory, with observations ordered by frame index:

1. Let `(u₀, v₀)` be the normalized center of the **first** observation and `(u₁, v₁)`
   the normalized center of the **last** observation. Observations in between are ignored.
2. **Stationary filter.** If `sqrt((u₁ − u₀)² + (v₁ − v₀)²) < 0.05`, the trajectory
   contributes **nothing**.
3. **Direction.** With counting line `L` (default `0.5`):
   - `u₀ < L` and `u₁ ≥ L` → one **rightward** passage
   - `u₀ ≥ L` and `u₁ < L` → one **leftward** passage
   - otherwise → no contribution
4. Each trajectory contributes **at most one** passage.

### Parameters

| Parameter | Value in `cfc-compatible-v1` | Notes |
|---|---|---|
| `line_x_normalized` | `0.5` | Required for benchmark-compatible evaluation. Other values are allowed for operational jobs, but the report then labels the result non-benchmark |
| `min_displacement_normalized` | `0.05` | Must equal the official value for benchmark evaluation |

### Behavior this implies (intended, and tested)

| Situation | Contribution |
|---|---|
| Starts left of the line, ends right of it | 1 rightward |
| Starts right of the line, ends left of it | 1 leftward |
| Stationary fish (start and end within 0.05) | 0 |
| Approaches the line and retreats without crossing | 0 |
| **Crosses the line and returns to its starting side** | **0** — not two passages |
| Crosses, returns, and crosses again (ends on the opposite side) | 1, in the net direction |
| Has a missing-frame gap in the middle | Counted from its endpoints; the gap does not matter |
| Single-observation trajectory | 0 (zero displacement) |
| Starts exactly on the line (`u₀ = L`) | Treated as right of the line (`≥`) |
| Ends exactly on the line (`u₁ = L`) coming from the left | 1 rightward (`≥`) |

### Known quirks, kept deliberately for compatibility

- **The displacement is anisotropic.** It is measured after normalizing x by width and y
  by height separately. On a wide frame, a horizontal pixel movement counts for less than
  the same vertical pixel movement. This is reproduced exactly so that our counts match
  the benchmark.
- **Track fragmentation changes counts.** A fish whose track breaks near the line can
  produce zero passages (for example, a fragment from `0.30→0.49` and another from
  `0.51→0.80`), or it can still produce one if one fragment spans the line. This is why
  fragmentation is measured by its **effect on counts**, not only by tracking metrics.
- **Only endpoints matter.** A long detour is invisible to the rule. The review interface
  shows the full trajectory so that a human can catch such cases.

## 3. Upstream and downstream

Image directions become upstream and downstream **only** when orientation is configured:

| `upstream_direction` | rightward means | leftward means |
|---|---|---|
| `right` | upstream | downstream |
| `left` | downstream | upstream |
| not set | — | — (only rightward/leftward counts are reported) |

`net = upstream − downstream`. For uploads, the user confirms orientation on a preview
frame, and it is recorded in the job and in every export.

**CFC clips have no recorded orientation.** The CFC README documents an
`upstream_direction` metadata field, but the published metadata
(`fish_counting_metadata.tar.gz`, verified 2026-09-30) does not contain it for any of the
1,567 clips. Benchmark evaluation therefore uses image directions (rightward/leftward)
only, which is also what the official nMAE measures.

Horizontal-flip augmentation during training swaps left and right. Wherever direction
labels are used, they must be swapped too.

## 4. Completed versus provisional

The rule applies only to **completed** trajectories. A trajectory is completed when the
tracker terminates it or the recording ends. While processing is in progress, line
crossings can be shown as **provisional markers** (useful as evidence in review), but
they never contribute to a count until the trajectory is completed.

A trajectory cut off by the end of a clip is counted from the observations that exist.
The official benchmark does the same.

## 5. Evaluation metric

The metric is **directional normalized mean absolute error, per location**, following
the official evaluator:

```
nMAE(location) = Σ_clips ( |R̂ − R| + |L̂ − L| )  /  Σ_clips ( R + L )
```

Here `R` and `L` are the ground-truth rightward and leftward counts, and ground-truth
counts come from applying the **same policy** to the reference trajectories. Each location
is reported separately. The headline aggregate is the **macro-average across locations**.

**Deviation from the official code:** when a location has zero true passages, the
official evaluator raises a division-by-zero error. PassageWatch instead reports nMAE as
*undefined* for that group, and reports the absolute count error and false counts per
hour instead.

Before any PassageWatch result is trusted, our implementation must reproduce the
official nMAE values on the published CFC baseline tracking results (Stage 3).

## 6. Versioning rules

- `cfc-compatible-v1` is frozen once released. Fixing a bug in its *implementation* (code
  that does not match this document) is allowed. Changing the *rule* is not.
- A new rule — for example, one that uses intermediate crossings, a different stationary
  test, or a configurable minimum track length — becomes `passagewatch-v2` (or similar)
  and gets its own document section and its own test fixtures.
- Every result stores the policy version. Reports that use a non-benchmark line position or
  a non-CFC policy say so explicitly.
