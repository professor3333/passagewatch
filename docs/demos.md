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

The same release counts some recordings slightly differently on an Apple (arm64) CPU and on
an x86 CPU. Found while loading these demos, and traced on 2026-10-09:

| Demo | Development Mac (arm64) | Release image (linux/amd64) | Recorded evaluation | CFC reference |
|---|---|---|---|---|
| Clear | 11 / 0 | 11 / 0 | 11 / 0 (holdout, Mac) | 11 / 0 |
| Difficult | 2 / 0 | **3 / 0** | 2 / 0 (holdout, Mac) | 5 / 0 |
| Unfamiliar camera | **4 / 2** | 4 / 3 | 4 / 3 (test, Kaggle x86) | 4 / 2 |

**Each host reproduces its own evaluation.** The release image, pulled by the digest in its
manifest and run on a GitHub linux/amd64 runner, counts the kenai-channel demo exactly as the
recorded test evaluation did on Kaggle (x86). The Mac counts the two holdout demos exactly as
the holdout report, which was measured on the Mac. On the 36 clips of the demo's
kenai-channel day, the Mac's counts differ from the recorded test evaluation on 4 clips (3
closer to the reference, 1 further) and its track numbers on 11. The day's nMAE is 0.241 on
the Mac and 0.264 recorded.

**The cause is the letterbox resize.** `scripts/host_parity.py` records a fingerprint of
every stage. The workflow `host-parity.yml` runs it inside the release image, and
`scripts/compare_host_parity.py` compares two runs. On all three demos, on both hosts:

| Stage | Mac vs release image |
|---|---|
| Decoded frames | identical |
| Temporal encoding (frame, background difference, motion) | identical |
| Letterboxed network input (`cv2.resize`, bilinear) | **different**: 8–15% of pixels differ by 1 grey level (at most 2) |
| The same resize with OpenCV's SIMD code turned off | still different |
| Detections, ONNX Runtime | scores differ by a median of about 6 × 10⁻⁴ and at most 0.18; boxes shift by up to 4.7 px |
| Detections, PyTorch (CPU), same checkpoint | the same differences as ONNX Runtime |
| ONNX Runtime vs PyTorch on the **same** host | scores within 1 × 10⁻⁵; same counts and tracks |

OpenCV's bilinear resize rounds differently on the two CPU architectures. The network
amplifies these one-level differences in some pixels into score changes that move a few
detections across the 0.4 threshold. That changes which detections a track links, and
occasionally a count. The runtimes are not the cause: two independent implementations agree
on each host and differ identically between hosts.

**What follows:**
- The service is deployed as the linux/amd64 image, the same architecture as the test
  evaluation, so the test results describe the deployed system.
- The development results in this repository (kenai-val, `kenai-holdout-v1`, the usability
  study's automatic counts) were measured on the Mac. On x86 they can differ slightly: on
  this day, 4 of 36 clips' counts.
- A resize that gives the same pixels on every CPU would remove the difference. It changes
  the network input, so it would be a new preprocessing version, evaluated again before a
  release. This is not done.

The fingerprints are in `releases/evaluations/passagewatch-0.3.0-host-parity/`. To reproduce
them, run the `host-parity` workflow, and on another host:

```bash
uv run python scripts/host_parity.py --archive dist/passagewatch-demos-v1.tar.gz \
    --bundle bundles/passagewatch-0.3.0 --out runs/host-parity/mac-m1.json
uv run python scripts/compare_host_parity.py runs/host-parity/mac-m1.json \
    releases/evaluations/passagewatch-0.3.0-host-parity/linux-amd64-image.json
```
