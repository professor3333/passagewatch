# Neural Baseline (YOLOX-Tiny, single frame)

The second system in the comparison: a COCO-pretrained **YOLOX-Tiny**, fine-tuned on sonar
frames, whose detections go through **the same Kalman tracker and counting policy as the
classical baseline**. Only the detector changes, so the comparison with `classical-v2` is
the plan's first one-factor ablation: classical versus neural proposals.

Code: `passagewatch.detection.neural`, `passagewatch.training`, `passagewatch.inference.neural`.
Training: [docs/training.md](training.md).

## Detector

| | |
|---|---|
| Model | YOLOX-Tiny (vendored at commit `6ddff48`), one class, COCO-pretrained (`yolox_tiny.pth`, SHA-256 `9de513de…`) |
| Input | `letterbox-gray3-v1`: 960 × 416, grayscale repeated to 3 channels, values 0–255 |
| Training data | kenai-train clips of `kenai-dev-v1` (183 clips, 5 days), every 3rd frame: 19,064 samples per epoch, including frames without fish |
| Schedule | 1 head-only epoch, then 29 epochs of full fine-tuning (backbone 0.1× LR); SGD, warm-up and cosine decay; batch 16; AMP |
| Augmentation | flip, contrast/brightness, noise, slight blur |
| Hardware | Kaggle, one NVIDIA Tesla T4: about 65 images/s, 30 epochs in about 2.5 h |
| Run | `configs/training/yolox-tiny-v1.yaml` at commit `681a02e`, manifest `full-v2` (content SHA-256 `a2ecbced…`) |

Training loss (mean per epoch) fell from 9.82 (epoch 1, head only) to 7.36 (epoch 2), 4.15
(epoch 10), 2.98 (epoch 20) and 2.50 (epoch 30).

## Selection protocol

- **Tracker and counting:** identical to `classical-v2`, from the `tracker` section of
  `configs/tracking/classical-v2.yaml` (gate 0.5 m, max_age 4, min_hits 3, min_length 8)
  and `cfc-compatible-v1`.
- **Candidates:** a declared shortlist of epochs 20, 25 and 30, where the training loss had
  levelled off, × detection score thresholds 0.1–0.5. Detection is about 20 minutes per
  epoch on the development Mac, so not all 30 epochs were evaluated.
- **Selection:** the lowest counting nMAE on **kenai-val**, as the project plan requires
  (released pipelines are selected by counting metric on official validation). The neural
  number on kenai-val is therefore a selected result and slightly optimistic, while
  `classical-v2` was selected on kenai-train and only measured on kenai-val. An unbiased
  comparison of both comes from the untouched test locations in Stage 11.

## Results

Measured on 2026-10-01 on all 64 kenai-val clips (183 true passages; full-length clips
from `kenai-dev-v1`), with tracker and counting as above.

### Selection grid (counting nMAE on kenai-val)

| Epoch | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.164 | 0.126 | 0.153 | 0.142 | 0.131 |
| 25 | 0.126 | **0.120** | 0.142 | 0.131 | 0.153 |
| 30 | 0.131 | 0.126 | 0.137 | 0.137 | 0.137 |

**Selected: epoch 25, score threshold 0.2** (checkpoint SHA-256 recorded in the report).
All 15 candidates beat `classical-v2`, the worst at 0.164, so the improvement does not
depend on the selection. Detection recall at IoU ≥ 0.3 is 0.67, with precision 0.86, against
0.23 / 0.40 for `classical-v2`.

### Comparison (paired bootstrap over the 64 clips, 10,000 resamples, 95% CI)

| System | kenai-val nMAE [95% CI] |
|---|---|
| `classical-v2` (tuned on kenai-train) | 0.235 [0.181, 0.300] |
| **YOLOX-Tiny, epoch 25, threshold 0.2** (selected on kenai-val) | **0.120 [0.078, 0.171]** |
| CFC published Baseline | 0.049 [0.018, 0.090] |
| CFC published Baseline++ | 0.033 [0.006, 0.070] |

| Difference | Estimate [95% CI] |
|---|---|
| YOLOX-Tiny − `classical-v2` | **−0.115 [−0.182, −0.049]**: better in 99.9% of resamples |
| CFC Baseline − YOLOX-Tiny | −0.071 [−0.105, −0.041]: CFC better in more than 99.9% of resamples |
| CFC Baseline++ − YOLOX-Tiny | −0.087 [−0.127, −0.052] |

`scripts/compare_counts.py` computes these intervals from the per-clip counts.

**Where the 22 remaining errors come from:** 17 missed passages and 5 false ones. Like the
classical system, the neural one errs mainly by missing fish.

**Why it is still behind CFC's baselines.** These are hypotheses, none of them tested yet:
- a smaller detector (YOLOX-Tiny; CFC trained YOLOv5m);
- a third of the training data (5 of 15 kenai-train days, every 3rd frame);
- a tracker tuned for the classical detector's output (`min_length` 8, `max_age` 4);
- CFC Baseline++'s temporal input channels.

Testing them is the job of Stages 5 and 8, which change one factor at a time.

### Unbiased check on `kenai-holdout-v1` (Stage 8)

The internal holdout (`docs/dataset_card.md`) is five kenai-train days that neither system
trained on or was selected on: 174 clips, 580 passages. Each system was evaluated once with
its fixed, already chosen settings (paired bootstrap over the 174 clips, 10,000 resamples,
95% CI):

| System | kenai-val nMAE | kenai-holdout-v1 nMAE |
|---|---|---|
| `classical-v2` | 0.235 [0.181, 0.300] | **0.369 [0.316, 0.430]** |
| YOLOX-Tiny, epoch 25, threshold 0.2 (released) | 0.120 [0.078, 0.171] | **0.210 [0.166, 0.261]** |
| YOLOX-Tiny − `classical-v2` | −0.115 [−0.182, −0.049] | **−0.159 [−0.224, −0.093]** |

On the holdout, the neural system has 99 missed and 23 false passages (detection recall 0.71,
precision 0.80); `classical-v2` has detection recall 0.30.

What this shows:

- **kenai-val flatters both systems.** Its single day (2018-06-03) is easier than these
  five days. `classical-v2` was never selected on kenai-val, yet it also gets worse (0.235 →
  0.369). So most of the gap is a day effect, not only the optimism of having selected the
  neural setting on kenai-val.
- **The neural system's advantage holds** on unseen days, and is larger there.
- **Day-to-day variation is large.** Single-day validation numbers should not be quoted as
  the system's accuracy. The holdout numbers are still from the same two cameras and the
  same season; the official test locations (Stage 11) measure transfer to other cameras
  and rivers.

### Runtime on the development Mac (Apple M1)

| Device | YOLOX-Tiny at 960 × 416, batch 8 |
|---|---|
| Apple GPU (MPS), kenai-val end to end with JPEG decoding | 28–31 ms/frame |
| CPU, model only | 247 ms/frame |

The serving target is CPU. At 247 ms per frame, a 10-minute recording at 8 frames/s would
take about 20 minutes. Stage 10 profiles this and tries ONNX Runtime, a smaller input size
and INT8, rerunning the full counting evaluation each time.

### Tracker experiment (Stage 5): Kalman tracker versus ByteTrack

The detections are identical (epoch 25, cached on kenai-val), and so are counting and
evaluation; only the tracker changes. ByteTrack (`passagewatch.tracking.bytetrack`) is
implemented from its paper, and associates by box IoU in two stages so that low-score
detections can extend existing tracks. Each plan gives both trackers the same budget.

**Plan 1** (`configs/tracking/tuning/tracker-plan-1.yaml`): ByteTrack at its usual score
scale.

| Tracker | Setting | nMAE | Missed / false |
|---|---|---:|---|
| Kalman | `max_age` 4, `min_length` 3 / **4, 8 (current)** / 8, 8 | 0.120 (tie) | 14–17 / 5–8 |
| Kalman | `max_age` 8, `min_length` 3 | 0.142 | 13 / 13 |
| ByteTrack | `high_threshold` 0.4, `min_length` 3 | 0.142 | 24 / 2 |
| ByteTrack | other three settings | 0.153–0.164 | 26–29 / 1–2 |

ByteTrack minus Kalman (best of each): **+0.022 [−0.023, +0.064]**. ByteTrack missed more
passages. Reference boxes overlap themselves well from frame to frame (median IoU 0.69; only
1.8% below 0.2), so IoU gating was not the cause. The cause was score scale: new tracks
started only at a score ≥ 0.6, far above this detector's operating point of 0.2.

**Plan 2** (`tracker-plan-2.yaml`, declared after seeing plan 1, so its result carries extra
selection): ByteTrack thresholds matched to the detector's score scale.

| Tracker | Setting | nMAE | Missed / false |
|---|---|---:|---|
| Kalman | current (`max_age` 4, `min_length` 8, detections ≥ 0.2) | 0.120 | 17 / 5 |
| ByteTrack | high 0.3, new track 0.4, `min_length` 3 | **0.109** | 18 / 2 |
| ByteTrack | high 0.2, new track 0.3, `min_length` 3 | 0.115 | 18 / 3 |
| ByteTrack | either, `min_length` 8 | 0.142 | 25 / 1 |

ByteTrack minus Kalman: **−0.011 [−0.048, +0.021]**, with ByteTrack better in 69% of
resamples.

**Decision: keep the Kalman tracker.** Calibrated ByteTrack is not measurably better (the
interval includes zero, and it was selected from a follow-up plan), and a retained change
needs a measured benefit. ByteTrack stays available as a tested alternative. Two things were
learned: ByteTrack's thresholds must match the detector's score scale, and with ByteTrack a
long minimum track length costs missed passages.

`scripts/compare_trackers.py` runs a plan on cached detections and prints these tables. On
ties it keeps the current settings.

### How to reproduce

```bash
uv run python scripts/evaluate_neural.py --run models/runs/yolox-tiny-v1 \
    --epochs 20 25 30 --thresholds 0.1 0.2 0.3 0.4 0.5 \
    --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1 --partition val
uv run python scripts/compare_counts.py \
    --a runs/classical/classical-v2-c4717f5f/full-v2/report.json \
    --b runs/neural/yolox-tiny-v1/report-val.json --b-epoch 25 --b-threshold 0.2
```

The checkpoints themselves (`models/runs/yolox-tiny-v1/`, 650 MB) are not committed.
Their versioning is the open question of [ADR 0001](decisions/0001-data-versioning.md).
