# Operations

How the service performs and behaves under load. Deployment and rollback are documented in
the README; this page collects measurements. Every number here was measured on the host it
names. None is a guarantee for other hardware.

## Performance profile (Stage 10)

`scripts/profile_pipeline.py` runs exactly the worker's path (`InferencePipeline` from a
release bundle, reading JPEG frames from an uploaded-style ZIP) on one recording. It runs once
to warm up, then reports time per stage and peak resident memory. The worker logs the same
stage times with every finished job (`stage_seconds` in the `job succeeded` log line).

**Host:** the development Mac, Apple M1 (4 performance + 4 efficiency cores, 8 GB RAM),
macOS, Python 3.12, CPU-only PyTorch path, measured on 2026-10-03. The machine was swapping
(about 6 GB of swap in use) during these runs, so treat the absolute numbers as indicative.

**Recording:** kenai-val clip `…LeftNear_Stratum1…_210000_2467_3008`, 541 frames of
626 × 835 px. **Release:** `passagewatch-0.2.0` (YOLOX-Tiny, 960 × 416 input, batch 8).

| Torch CPU threads | ms/frame | frames/s | Peak memory |
|---:|---:|---:|---:|
| 1 | 175 | 5.7 | 1.37 GB |
| 2 | 117 | 8.5 | 1.29 GB |
| **4** | **104** | **9.6** | 1.35 GB |
| 8 | 110 | 9.1 | 1.35 GB |

Where the time goes (4 threads):

| Stage | ms/frame | Share |
|---|---:|---:|
| decode (ZIP + JPEG) | 1.9 | 1.8% |
| preprocess (letterbox, tensor) | 1.0 | 0.9% |
| **detect (YOLOX forward)** | **100.7** | **97.0%** |
| postprocess (decode boxes, NMS) | 0.2 | 0.2% |
| track | < 0.1 | 0.0% |
| count | < 0.1 | 0.0% |
| review (`review-v0`, on `passagewatch-0.2.1`) | < 0.1 | 0.0% |

What this means:

- **The detector's forward pass is the only thing worth optimizing.** Decoding,
  preprocessing, tracking, counting and review scoring together take about 3% of the time.
- **Four threads is the best setting on this host.** The M1 has four performance cores;
  adding the efficiency cores makes it slightly slower. A deployment should set the worker's
  thread count to its performance-core count (`PASSAGEWATCH_TORCH_THREADS`) rather than rely
  on PyTorch's default.
- At about 10 frames/s, a recording is analyzed in roughly real time at CFC's typical 8–10
  frames/s. A 10-minute recording takes about 10 minutes.
- Peak memory is about 1.3 GB for this recording, most of it PyTorch and the model. Frames
  are streamed rather than held in memory; behavior on much longer recordings is checked
  separately (resource limits, below).

An earlier quick measurement in `docs/neural_baseline.md` (247 ms/frame, "CPU, model only")
was taken before this profiler existed, under unrecorded conditions; this profile supersedes
it.

The next optimization steps, in the order the project plan sets: ONNX Runtime, then
input-size trade-offs, then static INT8 with representative calibration frames. Each one
reruns the full counting evaluation on development data, with a budget of at most +1
percentage point of nMAE. They wait until the detector for the next release is final
(experiment 4).

Reproduce:

```bash
uv run python scripts/profile_pipeline.py --bundle bundles/passagewatch-0.2.0 \
    --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1 --threads 4 \
    --clip 2018-06-03-JD154_LeftNear_Stratum1_Set1_LN_2018-06-03_210000_2467_3008
```
