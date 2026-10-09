# Demo examples

Three recordings from the Caltech Fish Counting dataset show what PassageWatch does on a
clear case, a difficult one, and a camera it was never trained on. They are **illustrations,
not evidence**: the measured results are in [test results](test_results.md) and
[error analysis](error_analysis.md).

| Demo | Recording (CFC v1.1) | CFC reference → / ← | What it shows |
|---|---|---|---|
| **Clear** | `kenai-train` · `2018-06-06-JD157_LeftNear_Stratum1_Set1_LN_2018-06-06_190000_2060_2260` (200 frames) | 11 / 0 | A busy near-range recording that the release counts exactly |
| **Difficult** | `kenai-train` · `2018-06-02-JD153_LeftFar_Stratum2_Set1_LO_2018-06-02_001003_0_321` (321 frames) | 5 / 0 | Small, faint fish far from the sonar; the release misses three of five passages |
| **Unfamiliar camera** | `kenai-channel` · `2018-08-17-JD229_Channel_Stratum1_Set1_CH_2018-08-17_110006_385_585` (200 frames) | 4 / 2 | A test-location camera with fish moving both ways |

**How they were chosen.** By hand, from candidates that meet these rules, written down here
before the demos were published:

- **Clear and difficult** come from `kenai-holdout-v1`: recording days that no detector
  trained on and no setting was chosen on. None of them was used in the
  [usability study](usability_study.md).
  - **Clear:** counted exactly by the release (`runs/neural/yolox-tiny-t1/report-holdout.json`).
  - **Difficult:** the only holdout clip with a count error of 3 or more.
- **Unfamiliar camera** comes from kenai-channel, one of the four official test locations.
  The test set was evaluated once for `passagewatch-0.3.0` and is now a known benchmark, and
  nothing is tuned on it. The rule was a clip with passages in both directions and a count
  error close to the location's typical one. See **Results depend slightly on the host**
  below.

The reference counts are the CFC annotations counted with `cfc-compatible-v1`, computed by
`scripts/package_demos.py`.

## In the product

- **Home page.** When demos are loaded, the review interface lists them, each marked
  **cached result**.
- **Opening a demo.** "Open and review" creates the visitor's own job for the demo's
  recording. That job is a result-cache hit of the precomputed analysis, so it is ready at
  once and no new inference runs. The visitor's corrections stay in their copy and never
  change the shared example.
- **Job page.** A notice says the result is precomputed and cached, names the release that
  computed it and when, and gives the CFC reference counts.
- **Live analysis.** Uploading a recording always runs inference. Only a byte-identical
  recording analysed by the same pipeline is served from the cache, and it is labelled
  "cached result" too.
- **Retention and deletion.** A demo's recording never expires, and
  `DELETE /v1/clips/{id}` refuses it with `403`. The service recognises a demo by the
  SHA-256 of its frames ZIP in the catalog.
- **API.** `GET /v1/demos` lists the catalog with each demo's clip, precomputed job, the
  pipeline that computed it, and when. A demo is `available` once the active release has
  analysed it.

## Files

| | |
|---|---|
| `demos/catalog.json` | Committed. Each demo's text, CFC source, reference counts, sonar metadata, frame count and the SHA-256 of its frames ZIP |
| `passagewatch-demos-v1.tar.gz` | The frames, as reproducible ZIPs, with the catalog and license notices. Attached to the GitHub release [`demos-v1`](https://github.com/professor3333/passagewatch/releases/tag/demos-v1) |
| `configs/data/demo_channel.yaml` | Streams the one kenai-channel recording day the unfamiliar-camera demo needs |

The CFC data is MIT-licensed. The archive carries `LICENSE` and `THIRD_PARTY_NOTICES.md`,
which credit the dataset.

## Loading the demos

The image sets `PASSAGEWATCH_DEMO_CATALOG=/app/demos/catalog.json`. With the service
running:

```bash
uv run python scripts/load_demos.py --url http://127.0.0.1:8000
```

The script downloads the archive from the `demos-v1` release, checks every ZIP against the
catalog, and uploads each demo that is not loaded yet. It then waits for the active release
to analyse each one and prints the automatic counts next to the reference. Running it again
does nothing once all demos are loaded. After a new release is activated, run it again: the
demos are listed only once the new release has analysed them.

To rebuild the archive from the local CFC data (`make data-kenai-holdout`, and
`uv run python scripts/stream_subset.py --config configs/data/demo_channel.yaml`):

```bash
uv run python scripts/package_demos.py   # checks the catalog against the data, writes dist/
```

The same frames always give the same archive, byte for byte.

## Results depend slightly on the host

Loaded on the development Mac (Apple M1, ONNX Runtime on the CPU), release
`passagewatch-0.3.0` counts the demos as follows. The test evaluation ran on Kaggle (x86
CPU and an NVIDIA GPU).

| Demo | This Mac | Recorded evaluation | CFC reference |
|---|---|---|---|
| Clear | 11 / 0 | 11 / 0 (holdout report) | 11 / 0 |
| Difficult | 2 / 0 | 2 / 0 (holdout report) | 5 / 0 |
| Unfamiliar camera | **4 / 2** | **4 / 3** (test evaluation: PyTorch GPU and ONNX CPU) | 4 / 2 |

The unfamiliar-camera demo was chosen from the recorded evaluation (4 / 3, one false ←
passage). On this Mac it counts exactly. To see how far this goes, the release's worker path
was run on this Mac over all 36 clips of the same kenai-channel day (2026-10-09) and
compared with the recorded test evaluation:

| | |
|---|---|
| Clips with different counts | **4 of 36** (3 closer to the reference here, 1 further) |
| Clips with a different number of tracks | 11 of 36 |
| nMAE on this day (87 passages) | 0.241 on this Mac, 0.264 recorded |

Within one host the runtimes agree. On the Mac, PyTorch on the GPU, PyTorch on the CPU and
ONNX Runtime give the same counts on all 64 kenai-val clips. On Kaggle, ONNX Runtime on the
CPU and PyTorch on the GPU agree on all 20 test clips checked. The difference therefore
comes from something every runtime on a host shares, such as frame decoding or
preprocessing arithmetic. The cause has **not been established**, and the deployed image
(linux/amd64, like Kaggle) has not been checked against the recorded evaluation on these
clips. Small changes in boxes or scores can change which detections a track links, and so
a count. This is listed under the README's limitations and in [operations](operations.md).
