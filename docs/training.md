# Training a Detector

Training needs a GPU; everything else (development, evaluation, serving) runs on CPU.
PassageWatch trains on a free **Kaggle** notebook GPU. The notebook downloads the data
itself, so nothing is uploaded from the development machine.

## What a training run does

`tools/kaggle/train_on_kaggle.sh`, run from a fresh clone of the repository:

1. Creates a Python 3.12 environment with the locked dependencies (`uv.lock`) and the CUDA
   build of the **locked** torch and torchvision versions. It stops if no GPU is visible.
2. Downloads the CFC labels (MD5-verified) and streams the `kenai-dev-v1-train` frames
   (`configs/data/kenai_subset_train.yaml`: the five kenai-train days of `kenai-dev-v1`,
   about 12.5 GB kept from a 44 GB stream). Every frame is then checked against the
   committed SHA-256 inventory (`scripts/verify_frames.py`), so the GPU host trains on
   exactly the frames validated in manifest `full-v2`.
3. Trains with `scripts/train_yolox.py` (configuration `configs/training/yolox-tiny-v1.yaml`
   by default), resuming from `latest.pt` if the run directory already has one.

Only the kenai-train partition is used. kenai-val and the test locations are never loaded.

## One-time Kaggle setup

1. Sign in at <https://www.kaggle.com>.
2. **Verify your phone number** (Settings → Phone verification). Without it, notebooks
   cannot use a GPU or the internet.

## Running it

1. **Create** → **New Notebook**.
2. In the right-hand panel, under **Session options**:
   - **Accelerator:** *GPU T4 x2* (one GPU is used);
   - **Internet:** on.
3. Replace the first cell with:

   ```bash
   %%bash
   set -euo pipefail
   REF=main   # or a commit/tag; the run records the exact commit either way
   git clone --quiet https://github.com/professor3333/passagewatch.git /tmp/passagewatch
   cd /tmp/passagewatch
   git checkout --quiet "$REF"
   CONFIG=configs/training/yolox-tiny-v1.yaml bash tools/kaggle/train_on_kaggle.sh
   ```

   To train another configuration, change `CONFIG`. For example,
   `configs/training/yolox-tiny-v2.yaml` (Stage 8 experiment 3: the same training at a
   1280 × 640 input) has about twice the pixels per image, so expect training to take
   about twice as long. Its run directory is `runs/yolox-tiny-v2/`.
   `configs/training/yolox-tiny-t1.yaml` (experiment 4: temporal input at v1's input size)
   first computes each training clip's background once (a few minutes), then trains at
   about v1's speed. Its run directory is `runs/yolox-tiny-t1/`.

4. Click **Save Version** → **Save & Run All (Commit)** → **Save**. The notebook now runs
   in the background, so you can close the browser. Follow it under the notebook's
   **Versions** (the log shows streaming and training progress).

A healthy log goes through these stages, in order:

| Stage | What you see |
|---|---|
| Start | `== commit: <hash> …`, then a small table with the GPU name (e.g. `Tesla T4`) |
| Environment (a few minutes) | package installation, then `torch 2.14.1+cu126 on Tesla T4` |
| Data | the labels download, then `kenai-dev-v1-train: keeping 183 clips (57012 frames)` and progress lines such as `  12%  5.3/44.1 GB  9.2 MB/s  eta 70 min` |
| Check | `57012 files checked against 87530 inventoried` |
| Training | `yolox-tiny-v1: 19064 samples from 183 clips, 1192 iterations/epoch on cuda`, then a line every 50 iterations (`epoch 1/30  iter 50/35760  loss …  img/s`) and `finished epoch N/30 in … min` |
| End | `== done: /kaggle/working/runs/yolox-tiny-v1` and a file listing |

The session time limit and the weekly GPU quota are set by Kaggle and shown in your
account. Streaming the data also counts against the GPU quota, because the GPU session is
running.

## Collecting the results

When the version finishes, open it and go to **Output**. The run directory
`runs/yolox-tiny-v1/` contains:

| File | Contents |
|---|---|
| `epoch-NNN.pt` | Weights after each epoch, with config and metadata |
| `latest.pt` | Everything needed to resume (weights, optimizer, AMP scaler, RNG states, position) |
| `metrics.jsonl` | Losses, learning rate and throughput every 50 iterations |
| `run.json` | Config, seeds, sample counts, device, torch version, git commit, manifest hash |

Download the directory and put it at `models/runs/yolox-tiny-v1/` in your local clone
(`models/` is not committed). The next step chooses the released epoch by **counting
nMAE on kenai-val** through the full pipeline, not by training loss.

## If a run stops early

Checkpoints are written every 500 iterations and after every epoch, so a run that hits the
session limit loses at most a few minutes of work. To continue:

1. Open the notebook, click **Add Input**, and add **the output of the stopped version**
   (search for your notebook by name under *Your Work*).
2. Add a line `export RESUME_FROM=/kaggle/input/<input name>/runs/yolox-tiny-v1` before
   the `bash tools/kaggle/train_on_kaggle.sh` line, using the input's path as shown in the
   right-hand panel.
3. **Save Version** again. Training continues exactly where it stopped: the sample order and
   augmentation of each epoch are seeded, and the RNG states are restored.

## Training locally (smoke tests only)

The development Mac (Apple M1) can run a few iterations on its GPU through PyTorch's MPS
backend. It manages about 5 images per second, far too slow for a real run (an epoch
would take over an hour), but enough to check the pipeline:

```bash
uv run python scripts/train_yolox.py --config configs/training/yolox-tiny-v1.yaml \
    --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1 \
    --out runs/train/smoke --device mps --max-iters 20
```

## Training design

See the docstring of `passagewatch.training.yolox_train` for details.

- **Transfer learning:** COCO-pretrained YOLOX with a new one-class head. One head-only
  epoch, with the backbone and neck frozen (including their BatchNorm statistics), is
  followed by full fine-tuning in which the backbone and neck learn 10× slower than the
  head. `pretrained: null` gives the from-scratch control.
- **Learning rate:** YOLOX's linear scaling (0.01 / 64 per image), a quadratic warm-up
  over one epoch, then cosine decay to 5% of the peak. Training uses SGD with Nesterov
  momentum and weight decay on weights only.
- **Data:** every 3rd frame of each training clip, including frames without fish. Boxes
  are clipped to the frame. Preprocessing is the serving code itself
  (`letterbox-gray3-v1`, 960 × 416 input); a test checks that, with augmentation off, a
  training sample equals the serving input exactly.
- **Augmentation (modest):** horizontal flip, contrast/brightness, Gaussian noise and
  slight blur. There are no rotations, crops or color changes. Detection labels carry no
  direction of travel, so a flip has no left/right label to swap.
- **Temporal input (optional, `preprocessing: letterbox-temporal3-v1`):** three channels
  per frame: the frame, the frame minus the clip's mean background, and the motion to the
  next frame. The design follows the CFC authors' Baseline++; the docstring of
  `passagewatch.preprocessing.temporal` lists how it differs. Each clip's background is
  computed once and cached under `data/cache/backgrounds/` (`--background-cache`).
  Contrast and brightness changes are applied to the frames *and* the background, so the
  background-subtracted channel stays consistent; a flip is applied to the finished
  3-channel image and its boxes. The first layer keeps its COCO weights, with the three
  channels in place of red, green and blue. Re-initializing it instead is a separate
  experiment, not an assumption. Tests check that training samples equal the serving input
  and that the worker and offline evaluation encode clips the same way.
- **Reproducibility:** fixed seeds for Python, NumPy and PyTorch, plus a seeded per-epoch
  sample order. Augmentation is seeded per (seed, epoch, sample). A resumed run is
  bit-identical to an uninterrupted one on CPU (tested).
- **Not yet:** MLflow tracking. Metrics go to `metrics.jsonl` and `run.json`, which a later
  stage can import.

## Evaluating a release on the test locations (Kaggle)

The official test locations (kenai-rightbank, kenai-channel, elwha, nushagak) are evaluated
**once, for a declared release**, with everything frozen. Their images (87 GB) do not fit on
the development machine, so the evaluation runs on Kaggle with
`tools/kaggle/evaluate_release_on_kaggle.sh`:

- The release bundle you upload must match its committed manifest
  (`releases/manifests/<release>.json`), or nothing runs.
- Each location's archive interleaves all its clips, so it is streamed in chunks of whole
  recording days that fit the disk. Frames are MD5-checked and SHA-256-inventoried, then
  evaluated and deleted.
- For every clip it records the counts of the release (PyTorch on the GPU), the baseline
  release `passagewatch-0.2.0` and the classical pipeline. For a few clips per location it
  also records the release's ONNX export on the CPU, which is the deployed runtime, to
  confirm the counts match.
- Nothing is chosen: no epoch, threshold or setting is looked at or changed.

The script was rehearsed on 8 `kenai-holdout-v1` clips. Its release, baseline, classical and
ONNX counts equalled the holdout reports exactly.

**One-time setup: upload the release bundles.**

1. On Kaggle: **Datasets** → **New Dataset** → upload `passagewatch-release-bundles.zip` (it
   contains the folders `passagewatch-0.3.0/` and `passagewatch-0.2.0/`, about 61 MB).
2. Give it a title such as `passagewatch-release-bundles`, keep it **Private**, and
   **Create**.

**Session 1: kenai-channel, nushagak and elwha** (about 36 GB to stream).

1. **Create** → **New Notebook**; **Accelerator:** GPU T4 x2; **Internet:** on.
2. **Add Input** → **Your Datasets** → `passagewatch-release-bundles`. The right-hand panel
   shows its path (for example `/kaggle/input/passagewatch-release-bundles`).
3. Replace the first cell with the following. Use the path from step 2 for `BUNDLES`, and the
   folder that contains `passagewatch-0.3.0/`:

   ```bash
   %%bash
   set -euo pipefail
   git clone --quiet https://github.com/professor3333/passagewatch.git /tmp/passagewatch
   cd /tmp/passagewatch
   export BUNDLES=/kaggle/input/passagewatch-release-bundles
   export LOCATIONS="kenai-channel nushagak elwha"
   bash tools/kaggle/evaluate_release_on_kaggle.sh
   ```

4. **Save Version** → **Save & Run All (Commit)**. The log shows
   `== chunk budget`, then for each location the streaming progress and
   `<location>: N clips evaluated, 0 quarantined`.
5. When it finishes, download the output (`release-eval/`).

**Session 2: kenai-rightbank** (51 GB per streaming pass). Use the same notebook with
`LOCATIONS="kenai-rightbank"` and save a new version. If a session stops early, add the
stopped version's output as an input and set
`export RESUME_FROM=/kaggle/input/<that output>/release-eval`. Clips already evaluated are
skipped.

Back on the development machine, the two downloads' `results/` files go into one directory,
and `scripts/summarize_release_evaluation.py --results <dir>` computes the per-location and
macro-average results. They are recorded in `releases/evaluations/`.
