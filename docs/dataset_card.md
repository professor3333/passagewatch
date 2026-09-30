# Dataset Card: Caltech Fish Counting (CFC)

> **Status: Stage 2, in progress.** This card records what has been verified so far.
> Validation results, the split manifest, and the known annotation issues are added as
> they are measured.

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
- **Some boxes extend beyond the image:** the smallest `bb_top` is `−1`, and at least one
  box extends past the right or bottom edge. How these are handled will be defined by the
  validator.

## The tiny subset

`tiny_dataset.tar.gz` (1.44 GB) contains **20 clips from each of the six locations, with 50
consecutive frames per clip** (6,000 frames), plus `gt_tiny.txt` annotations and subset
metadata. It was produced by CFC's `CFC/tools/get_tiny_dataset.py`.

- **80 of its 120 clips come from the official test locations.** PassageWatch uses those
  clips only for pipeline development (checking that files load and display). They are
  never used for training, tuning, threshold selection, or error mining.
- **The annotations and images are not aligned one to one.** `gt_tiny.txt` keeps the rows
  for the first 50 *annotated* frame numbers (1-based), while the image window is chosen by
  comparing those numbers with 0-based file numbers. As a result, some annotations
  reference frames whose images are not in the subset, and some images have no
  annotations. For example, one elwha clip has images `197–246` but annotated frames
  `209–255`. The validator must report this, and tiny-subset annotations must never be
  assumed complete for a frame.
- **Tiny-subset counts are not benchmark counts.** Trajectories are truncated to 50
  frames, so counting results on the tiny subset cannot be compared with published
  benchmark numbers.

## Known limitations

- Clips were selected around known fish activity. Results on CFC do not establish
  false-alarm rates on continuous, mostly empty footage.
- Seven sonar cameras across three rivers: limited geographic and hardware coverage.
- Adjacent frames are highly correlated. Splits are made by whole clip, never by frame.
- Annotation errors are possible. The label space supports fish detection, not species
  identification.
