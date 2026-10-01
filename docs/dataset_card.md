# Dataset Card: Caltech Fish Counting (CFC)

> **Status: complete for Stage 2** (tiny subset downloaded; full imagery not yet). Every
> number below was computed by PassageWatch from the published files on 2026-09-30 and
> 2026-10-01, unless it is attributed to the paper.

## Source and license

| | |
|---|---|
| Name | Caltech Fish Counting Dataset (CFC) |
| Publisher | California Institute of Technology, via CaltechDATA |
| Records | v1.0 [`1y23m-j8r69`](https://data.caltech.edu/records/1y23m-j8r69) (annotations, metadata, tiny subset, baseline results); v1.1 [`g945x-41103`](https://data.caltech.edu/records/g945x-41103) (per-location imagery, YOLO/COCO labels) |
| License | MIT (record metadata `rights: mit`; license text in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)) |
| Paper | Kay et al., *The Caltech Fish Counting Dataset: A Benchmark for Multiple-Object Tracking and Counting*, ECCV 2022 |
| Modality | Single-channel sonar imagery (ARIS/DIDSON renderings), stored as ordered JPEG frames |
| Label space | One class, *fish*, as bounding boxes with per-clip track identities. **No species labels.** |

Every downloadable file is listed with its size and MD5 in
[`configs/data/cfc_sources.yaml`](../configs/data/cfc_sources.yaml). Downloads are
rejected unless they match. Every extracted file's SHA-256 is recorded in
[`data/manifests/inventory/cfc/`](../data/manifests/inventory/cfc/).

## Composition

These statistics were computed by PassageWatch from the published MOT annotations and clip
metadata (v1.0 record) on 2026-09-30. "Tracks" is the sum over clips of the distinct track
IDs in each clip.

| Location | Official split | Clips | Frames | Boxes | Tracks | Frame rates (fps) | Frame sizes (W×H, most common) |
|---|---|---:|---:|---:|---:|---|---|
| kenai-train | train | 482 | 162,680 | 132,220 | 1,762 | 6.69, 8.0, 9.0, 10.0 | 286×622, 288×624, 288×623 (+72 more) |
| kenai-val | validation | 64 | 30,518 | 18,565 | 207 | 8.0, 9.0, 10.0 | 626×835, 622×830, 286×620 (+15 more) |
| kenai-rightbank | **test** | 657 | 199,965 | 171,416 | 2,434 | 6.6, 6.7, 9.0 | 513×882, 515×886, 288×624 (+86 more) |
| kenai-channel | **test** | 69 | 13,159 | 41,982 | 381 | 8.0 | 217×486, 427×826, 216×485 (+11 more) |
| elwha | **test** | 223 | 99,293 | 40,510 | 398 | 7.51, 8.35 | 632×1256, 630×1253, 628×1248 (+17 more) |
| nushagak | **test** | 72 | 21,600 | 111,240 | 3,070 | 13.33 | 1088×2129, 1084×2120, 1086×2124 (+4 more) |
| **Total** | | **1,567** | **527,215** | **515,933** | **8,252** | | |

The paper reports about 527,000 frames, about 516,000 boxes, and 8,254 tracks. Our track
count differs by 2; the cause has not been investigated yet. Summing `num_frames /
framerate` gives about 17.1 hours, while the paper states about 16.7 hours.

**Frame size varies from clip to clip, even within one location**, because the rendered
sonar window changes with range settings. Preprocessing therefore cannot assume a fixed
input size for a location.

## Verified format facts

- **MOT rows:** `frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z`. Frames,
  IDs, and box coordinates are 1-indexed.
- **Frame files are 0-indexed:** gt frame `N` is image `N−1.jpg`. Across all clips, gt
  frames lie in `[1, num_frames]`, and 470 clips annotate frame `num_frames` itself.
- **The clip metadata has no `upstream_direction`**, even though the CFC README documents
  one. The fields that are present are `clip_name`, `num_frames`, `framerate`, `width`,
  `height`, and the meter extents `x_meter_start`, `x_meter_stop`, `y_meter_start`,
  `y_meter_stop`.
- **Some boxes extend beyond the image:** 14 boxes in 10 clips (smallest `bb_top` is `−1`,
  one box extends past the bottom edge). They are kept **unclipped**, as the official
  evaluator does, and reported as warnings.
- **`gt.txt` rows are sorted by frame** in every clip.
- **Clip names encode their source recording.** Every one of the 1,567 names is
  `<recording>_<start>_<stop>`: frames `[start, stop)` of a source recording whose name ends
  in `YYYY-MM-DD_HHMMSS`, and `stop − start = num_frames` in every clip. There are 1,021
  recordings. No recording appears in two locations, and no two clips of one recording
  overlap, but 32 pairs of clips are back to back (one ends where the next begins).
- **The frame conversion is confirmed from the images.** Fish are brighter than the water
  around them. On the 40 kenai-train and kenai-val tiny clips, the mean contrast between a
  box and a ring around it is highest when each box is compared with image `N−1` for gt
  frame `N` (33.1, against 31.0 at image `N` and 30.2 at image `N−2`), and this offset is
  the best one in 31 of the 40 clips. A slow test keeps checking this
  (`tests/data/test_viewer_and_alignment.py`). The viewer shows the same alignment by eye.

## The tiny subset

`tiny_dataset.tar.gz` (1.44 GB) contains **20 clips from each of the six locations, with 50
consecutive frames per clip** (6,000 frames), plus `gt_tiny.txt` annotations and subset
metadata. It was produced by CFC's `CFC/tools/get_tiny_dataset.py`.

- **80 of its 120 clips come from the official test locations.** PassageWatch uses those
  clips only for pipeline development (checking that files load and display). They are
  never used for training, tuning, threshold selection, or error mining.
- **`gt_tiny.txt` is not aligned with the images, so PassageWatch does not use it.** We
  verified the mechanism against the tool's source (`visipedia/caltech-fish-counting` commit
  `81380c9`) and reproduced it on all 120 clips:
  - `gt_tiny.txt` copies `gt.txt` rows in file order until it has seen 50 distinct frame
    numbers. It stops after the *first* row of the 50th frame, so that frame can be
    incomplete.
  - The image window starts at `min(annotated frame) − 12`, or at image `0` when that frame
    is 12 or earlier. The 1-based frame number is used as a 0-based file number, so the
    first annotation lands 11 images into the window, not 12. Annotated frames that are not
    consecutive run past the end of the window.

  For example, one elwha clip has images `197–246` but `gt_tiny.txt` frames `209–255`.
- **PassageWatch takes tiny-clip boxes from the full `gt.txt`, restricted to the image
  window.** Every tiny image then has its complete annotations. `gt_tiny.txt` is only
  checked to be a subset of `gt.txt` (it is, in all 120 clips). The windows contain 13,335
  boxes on 4,842 annotated frames.
- **Most tiny trajectories are truncated.** 269 of the 452 tracks inside the windows also
  have boxes outside them.
- **Tiny-subset counts are not benchmark counts.** Trajectories are truncated to 50
  frames, so counting results on the tiny subset cannot be compared with published
  benchmark numbers.
- **Four tiny clips have images one pixel larger or smaller than their metadata** (for
  example, images `789×1933` with metadata `789×1932`). Box bounds and the counting
  normalization use the metadata size, as the official evaluator does. These are recorded
  as warnings.

## Validation

`scripts/validate_data.py` (`make validate-tiny`) parses every clip, converts it to the
internal convention (`docs/counting_policy.md` §1), and checks it. Each problem gets an issue
code. A clip with any **error** is **quarantined**: it stays on disk and is listed with its
reasons, and later stages must not use it. A **warning** is recorded, and the clip stays
usable. Nothing is repaired or skipped silently. The reports are committed at
[`data/manifests/validation/cfc/`](../data/manifests/validation/cfc/):
`tiny.json` (with every frame decoded) and `full.json` (annotations and metadata only,
because the full frames have not been downloaded).

| Code | Severity | Meaning |
|---|---|---|
| `missing_metadata_file` | error | A location has no metadata file (reported per location) |
| `invalid_metadata` | error | A metadata entry fails the schema (unknown field, non-positive size or rate) or its `clip_name` is duplicated |
| `missing_metadata` | error | A clip directory has no valid metadata entry |
| `missing_annotations` | error | A clip has no `gt.txt` |
| `missing_clip_directory` | error | A metadata entry has no clip or frame directory |
| `mot_format` | error | A MOT row cannot be parsed (reported with file and line) |
| `frame_out_of_range` | error | An annotated frame is outside `[1, num_frames]` |
| `duplicate_track_frame` | error | A track has two boxes in the same frame |
| `non_positive_box_size` | error | Box width or height ≤ 0 |
| `box_outside_image` | error | A box does not overlap the image at all |
| `missing_frames` | error | A frame file is missing inside the expected window |
| `frame_window_out_of_range` | error | Frame files exist beyond `num_frames` |
| `unreadable_frame` | error | A frame cannot be decoded |
| `image_size_mismatch` | error | Image size differs from the metadata by more than 1 px |
| `tiny_rows_not_in_gt` | error | `gt_tiny.txt` has rows that are not in `gt.txt` |
| `invalid_clip_name` | error | The clip name is not `<recording>_<start>_<stop>`, or `stop − start ≠ num_frames` |
| `box_partially_outside_image` | warning | A box extends beyond the image; it is kept unclipped |
| `image_size_differs_slightly` | warning | Image size differs from the metadata by exactly 1 px |
| `unexpected_file` | warning | A stray file in a clip directory (for example `.ipynb_checkpoints`); ignored |
| `missing_tiny_annotations` | warning | A tiny clip has no `gt_tiny.txt` (it is not used anyway) |
| `no_boxes_in_window` | warning | A tiny window contains no boxes |

Results on 2026-09-30:

| Data | Clips | Quarantined | Warnings |
|---|---:|---:|---|
| Tiny subset (6,000 frames decoded) | 120 | 0 | 4 clips with 1 px size differences; 2 boxes partly outside (1 clip) |
| Full annotations + metadata | 1,567 | 0 | 14 boxes partly outside (10 clips); 1 stray `.ipynb_checkpoints` (kenai-train) |

Also recorded per clip, as statistics rather than issues: tracks with missing frames
between their first and last box (801 of 8,252 in the full annotations, which is normal for
occlusions), and single-box tracks (1 in the full annotations).

## Splits and manifests

**Rule.** A clip's partition is its location's official split: kenai-train → `train`,
kenai-val → `val`, and kenai-rightbank, kenai-channel, elwha, nushagak → `test`. Whole clips
are assigned; frames are never split. Test clips have `tuning_allowed = false`: they are
never used for training, tuning, threshold selection, or error mining, including the 80
test-location clips in the tiny subset. Quarantined clips stay in the manifest with
`usable = false`.

| Location | Partition | Clips | Recordings | Days | Dates |
|---|---|---:|---:|---:|---|
| kenai-train | train | 482 | 346 | 15 | 2018-05-26 … 2018-06-10 |
| kenai-val | val | 64 | 45 | 1 | 2018-06-03 |
| kenai-rightbank | test | 657 | 390 | 16 | 2018-05-26 … 2018-06-10 |
| kenai-channel | test | 69 | 62 | 2 | 2018-08-16 … 2018-08-17 |
| elwha | test | 223 | 167 | 23 | 2018-07-09 … 2018-09-14 |
| nushagak | test | 72 | 11 | 7 | 2018-07-02 … 2018-08-06 |

What these groups mean for later claims:

- **Train and val share no day.** Validation is a held-out day (2018-06-03), but that day
  lies inside the training period. It measures generalization to new recordings and a new
  day, not to a later season.
- **kenai-rightbank was recorded on the same 16 days as kenai-train and kenai-val**, by a
  sonar on the other bank. Results on it measure transfer to a new camera position on the
  same river and days, not to a new time period.
- **Internal holdouts** (for example, the data used to fit a confidence calibrator) must be
  carved out of `train` by `recording_date` or at least by `recording_id`, never by clip.
  Back-to-back clips from one recording must stay together.

**Manifests.** `scripts/build_manifest.py --version <subset>-v<N>` (`make manifest-tiny`)
validates the clips and writes `data/manifests/splits/cfc/<version>.parquet` plus a JSON
sidecar. A version is immutable: rerunning with identical content is a no-op, and any
difference is refused until a new version name is used. The sidecar records the content
hash of the rows, the SHA-256 of the validation report and inventories it came from, and
per-partition totals. `read_manifest` checks the content hash on every read. Committed
versions:

| Version | Clips | Usable | Frames validated | Notes |
|---|---:|---:|---|---|
| `tiny-v1` | 120 | 120 | yes (6,000 frames) | train 20, val 20, test 80 clips |
| `full-v1` | 1,567 | 1,567 | no | annotations and metadata only; the full imagery is not downloaded yet |

Columns (schema version 1):

| Column | Meaning |
|---|---|
| `location`, `clip_name` | Clip identity |
| `recording_id`, `recording_date`, `recording_frame_start`, `recording_frame_stop` | Source recording and the clip's frame range in it, parsed from the clip name |
| `official_split`, `partition` | Publisher split, and the partition PassageWatch uses (identical in v1) |
| `tuning_allowed` | `false` for test locations |
| `status`, `usable`, `quarantine_reasons`, `warning_codes` | Validation outcome (see [Validation](#validation)) |
| `frames_validated` | Whether the frames were decoded and checked |
| `num_frames`, `width`, `height`, `framerate` | Clip metadata |
| `frame_start`, `frame_stop` | The frame window available (the whole clip, or the tiny window) |
| `boxes`, `tracks` | Annotation totals inside the window |
| `gt_sha256` | SHA-256 of the clip's `gt.txt` |

Data versioning uses these hash-chained manifests rather than DVC; see
[ADR 0001](decisions/0001-data-versioning.md).

**Viewing a clip.** `scripts/view_clip.py --location <loc> --clip <name>` writes an MP4 of
the clip window with boxes, track IDs, the counting line at `x = 0.5`, and the frame index
and time. `--sheet` writes a PNG contact sheet instead, and `--list` lists the clips.
Output goes to `data/cache/viewer/`, which is not committed.

## Known limitations

- Clips were selected around known fish activity. Results on CFC do not establish
  false-alarm rates on continuous, mostly empty footage.
- Seven sonar cameras across three rivers: limited geographic and hardware coverage.
- Adjacent frames are highly correlated. Splits are made by whole clip, never by frame.
- Annotation errors are possible. The label space supports fish detection, not species
  identification.
