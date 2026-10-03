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
- Peak memory is about 1.3 GB, most of it PyTorch and the model, and it does not grow with
  the recording's length (below).

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

## Resource limits on long recordings

The service accepts uploads of up to 6000 frames (`PASSAGEWATCH_MAX_FRAMES`). Frames are
streamed from the upload, and only detections and trajectories are kept, so memory should
not grow with length. Measured with `profile_pipeline.py --frames 6000`, which cycles the
same clip's frames (same host and release as above, 4 threads, no warm-up):

| Recording | Peak memory | ms/frame | Total |
|---|---:|---:|---:|
| 541 frames | 1.35 GB | 104 | 56 s |
| 6000 frames (the upload limit) | 1.36 GB | 91 | 546 s (9.1 min) |

Peak memory is flat, well under the design's 4 GB budget for the worker, and the longest
allowed recording finishes in about 9 minutes, inside the default one-hour job deadline
(`PASSAGEWATCH_JOB_DEADLINE_SECONDS`). Temporal preprocessing reads each recording twice
(background, then encoding), which adds decode time (about 2% of the total here) but no
memory.

## Reliability (Stage 10)

What the tests prove on every CI run, without the real model (a stand-in detector finds the
synthetic fish):

| Failure | Behavior | Test |
|---|---|---|
| Worker killed (SIGKILL) mid-job | The lease expires; another worker reruns the job; exactly one result | `test_a_worker_killed_mid_job_is_recovered_without_duplicates` |
| Worker loses its lease mid-job | It stops without publishing | `test_a_worker_that_loses_its_lease_stops_without_publishing` |
| Unexpected error | Retried until `max_attempts`, then failed | `test_unexpected_errors_are_retried` |
| Corrupt frame | Failed at once, not retried | `test_a_corrupt_frame_fails_the_job_permanently` |
| Job over its deadline | Failed | `test_job_past_its_deadline_fails` |
| Two workers leasing at once | Each job goes to one worker | `test_concurrent_workers_never_lease_the_same_job` |
| **100 jobs, two workers, a third failing once** (before artifacts, between artifacts and publication, or just after publication) | All 100 succeed; each has exactly one automatic result revision, one artifact directory and the undisturbed counts; a crash after publication is not retried | `test_retries_and_concurrent_workers_never_duplicate_results` |

The last test covers the design target of at least 99% completion over 100 valid jobs with
zero duplicated results after retries. With a stand-in detector it shows that the job
machinery behaves; it does not show that the real model never fails.
