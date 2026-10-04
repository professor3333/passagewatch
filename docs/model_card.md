# Model Card: PassageWatch fish detector and counting pipeline

This card describes the pipeline of release **`passagewatch-0.3.0`**. That release uses the
detector `yolox-tiny-t1`, epoch 25, at score threshold 0.4, with the `classical-v2` Kalman
tracker, counting policy `cfc-compatible-v1` and calibration version `review-v0`. All results
below come from development data. The official test locations are evaluated once, for a
declared release, in Stage 11; until then no result on them exists.

## Intended use

Detecting and tracking fish in sonar video (ARIS-style imaging sonar, as in the Caltech Fish
Counting dataset) to produce **suggested** directional passage counts. A fisheries technician
reviews and corrects them before export. Every count comes with its trajectories, evidence
frames and a review state; automatic and reviewed counts are kept separate.

## Not for

- Species identification. The model knows only "fish" versus "not fish".
- Identifying individual fish across recordings. A track ID is a trajectory within one
  analyzed recording.
- Unreviewed counts used as final figures.
- Claims about false-alarm rates on continuous, mostly empty footage. The training and
  evaluation clips were selected around fish activity.

## Provenance

| Item | Value |
|---|---|
| Architecture | YOLOX-Tiny, one class, vendored from <https://github.com/Megvii-BaseDetection/YOLOX> at commit `6ddff48` (Apache-2.0; see `THIRD_PARTY_NOTICES.md`) |
| Initialization | COCO-pretrained `yolox_tiny.pth` (release 0.1.1rc0, SHA-256 `9de513de…`), new one-class head |
| Input | `letterbox-temporal3-v1`, 960 × 416: the frame, the frame minus the recording's mean background, and the motion to the next frame |
| First layer | Keeps its COCO weights, with the three channels in place of red, green and blue. Re-initializing it was not tested |
| Temporal input credit | The Caltech Fish Counting authors' **Baseline++** (Kay et al., ECCV 2022, `CFC/convert.py`). Differences: fixed scaling instead of each clip's largest difference, motion from blurred frames, and the last frame looks back instead of being dropped (`passagewatch.preprocessing.temporal`) |
| Training run | `configs/training/yolox-tiny-t1.yaml`, Kaggle Tesla T4, commit `8341d37`, 30 epochs; released checkpoint epoch 25 (SHA-256 `3230a03c…`) |
| Tracker | `configs/tracking/classical-v2.yaml` (constant-velocity Kalman filter, 0.5 m gate), tuned on kenai-train for the classical baseline |
| Serving runtime | ONNX Runtime 1.30 on the CPU, from an ONNX export of the checkpoint (opset 17, SHA-256 `adb448af…`). On kenai-val it gives the same counts and track numbers as PyTorch on the CPU and on Apple's GPU, on all 64 clips |

## Training data

- **Caltech Fish Counting (CFC)**, record `g945x-41103` (MIT license); see
  `docs/dataset_card.md`.
- Manifest `full-v2` (content SHA-256 `a2ecbced…`), partition `train`: 183 kenai-train
  clips from five recording days (`kenai-dev-v1`), every third frame (19,064 samples).
- Split design: the publisher's train/val/test separation is kept, whole clips only, never
  frames. kenai-val (one day) selects settings. `kenai-holdout-v1` (five other kenai-train
  days, no shared day) confirms them. The four test locations (kenai-rightbank,
  kenai-channel, elwha, nushagak) are never used for training or tuning.

## Metrics (development data)

Directional nMAE = Σ(|R̂ − R| + |L̂ − L|) / Σ(R + L), with paired clip-bootstrap 95% intervals.

| Data | Clips | Passages | `classical-v2` | YOLOX-Tiny v1 | **This release** |
|---|---:|---:|---|---|---|
| kenai-val (selection) | 64 | 183 | 0.235 | 0.120 | **0.066** [0.027, 0.111] |
| kenai-holdout-v1 (confirmation) | 174 | 580 | 0.369 | 0.210 | **0.117** [0.089, 0.149] |

- Holdout: 44 missed and 24 false passages; detection recall 0.79 and precision 0.88
  (IoU ≥ 0.3, score ≥ 0.4).
- Small-target recall (kenai-val, by reference box area): 0.40 below 0.02 m², 0.51 at
  0.02–0.05 m², 0.79 at 0.05–0.1 m², 0.86 at ≥ 0.1 m².
- Passage-level errors on kenai-val: 13 (2 fish missed outright, 5 merged, 2 split,
  1 ambiguous start/end, 2 duplicates, 1 background, 1 non-passing fish). See
  `docs/error_analysis.md`.
- Not measured yet: AP50, AP50:95, HOTA, IDF1 and fragmentation. Counting quality was the
  selection criterion, as the project plan requires.
- Calibration: none. The **review score is heuristic** (`review-v0`, `docs/review.md`),
  shown as a ranking aid, never as a probability. No count confidence intervals are shown in
  the product.

The kenai-val number is a selected result; the holdout number is the unbiased one for new
days of the same two cameras. Neither says how the model transfers to other cameras or
rivers.

## Known failure modes

- Small or faint fish (recall 0.40 below 0.02 m²).
- Sediment, debris and other moving returns, which can look like fish in the motion channel.
- Dense scenes: fish passing close together merge into one track.
- Track fragmentation near the counting line, which can turn one passage into zero.
- Short occlusions and gaps longer than the tracker's 4-frame limit.
- Unfamiliar cameras, sonar settings or rivers (untested until Stage 11).
- A wrong upstream configuration flips upstream and downstream; image directions are
  unaffected.
- Frame-timing or decoding problems: dropped or duplicated frames change tracking.

## Release process

- **Thresholds:** the detector's epoch and score threshold are chosen on kenai-val by
  counting nMAE over a declared grid, with a fixed tie rule. A change is adopted only if a
  rule written down before its results is met on `kenai-holdout-v1`. Review-state thresholds
  are chosen the same way (Stage 9).
- **Promotion:** a release is an immutable bundle (`scripts/build_bundle.py`) with the
  checkpoint's SHA-256, preprocessing version, tracker configuration, counting-policy and
  calibration versions, and its config hash. The service checks all of them at start-up and
  reports them at `/v1/model-info`. A regression test requires the service to count exactly
  like the offline evaluation of the release.
- **Rollback:** the `bundles/active` pointer is switched back to the previous bundle and the
  worker restarted. Existing jobs keep the versions recorded for them.
