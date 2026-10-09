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

## Container hardening (Stage 10)

`docker-compose.yml` runs the API and the worker from one image with:

| Measure | Setting |
|---|---|
| Health checks | API: `/health/live`. Worker: `python -m passagewatch.service.worker.health`, healthy while a worker process on the container heartbeats (it does so idle and mid-job, after its model has loaded) |
| Memory limits | API 512 MB (measured about 0.14 GB while serving uploads and frames); worker 3 GB (measured peak about 1.4 GB at the 6000-frame upload limit; design budget 4 GB) |
| Filesystem | Read-only root; only the `/data` volume and a `/tmp` tmpfs are writable; release bundles are mounted read-only |
| Privileges | Non-root user (image), all Linux capabilities dropped, `no-new-privileges` |
| Processes | An init process forwards signals; the worker gets 30 s to stop gracefully, and an interrupted job is recovered from its lease |
| Logs | JSON logs, rotated at 10 MB × 5 files per container |
| CPU threads | `PASSAGEWATCH_TORCH_THREADS` (empty = PyTorch's default; set it to the host's performance cores) |

The Compose smoke test in CI starts the stack with a random-weights bundle, waits for both
health checks, checks that both containers are read-only and without capabilities, and runs
a job end to end.

## ONNX Runtime (Stage 10)

`scripts/export_onnx.py` exports a checkpoint's network to ONNX (opset 17, fixed input size,
dynamic batch; YOLOX's grid decoding is part of the graph). `passagewatch.detection.onnx`
runs it with ONNX Runtime on the CPU behind the same interface as the PyTorch model, so
letterboxing, box decoding, NMS and the mapping back to frame pixels are the same code. A
CI test compares the two on a random-weights model; `evaluate_neural.py --onnx` reruns the
full counting evaluation with the exported network.

**Counting parity on kenai-val** (64 clips, 183 passages; released YOLOX-Tiny, epoch 25,
threshold 0.2):

| Runtime | nMAE | Clips with different counts | Clips with different track numbers |
|---|---:|---:|---:|
| PyTorch, Apple GPU (MPS): the evaluated release | 0.1202 | — | — |
| PyTorch, CPU: what the service runs today | 0.1202 | 0 of 64 vs MPS | — |
| ONNX Runtime 1.30, CPU | 0.1202 | 0 of 64 vs PyTorch CPU | 0 of 64 |

The raw network outputs differ by at most about 0.001 px in box coordinates and 1e-5 in
scores. The service's CPU pipeline therefore counts exactly like the evaluated release, and
ONNX Runtime keeps that, well inside the +1 point nMAE budget for optimizations.

**Speed** (development M1, CPU, 4 threads, batch 8): on the same 64 frames, timed in five
interleaved rounds, detection took a median **123 ms/frame with PyTorch and 103 ms/frame
with ONNX Runtime** (about 17% less). Two full kenai-val runs at different times gave 111
and 113 ms/frame including reading frames from disk, which is within the noise of a host
under memory pressure. ONNX Runtime is a modest gain, not a large one, on this host.

**Serving with ONNX Runtime.** A release bundle built with `build_bundle.py --onnx <export>`
declares `detector.runtime: onnxruntime` and the ONNX file's SHA-256 (`detector.onnx` in the
bundle). The worker checks that hash, runs ONNX Runtime on the CPU only (another device is
refused, never silently replaced), and uses `PASSAGEWATCH_TORCH_THREADS` for its threads.
PyTorch bundles are unchanged, and their config hashes do not include the new fields, so
existing releases keep their identity. Through the worker's own path, on the same 541-frame
clip with 4 threads and runs alternated, a bundle of the released checkpoint with its ONNX
export gave **the same counts and identical trajectories** as `passagewatch-0.2.0`, at
**89–98 ms/frame against 106–109 ms/frame** (about 12–16% faster end to end).

The active release still uses PyTorch. The next release (once experiment 4 has decided the
detector) is built with ONNX after rerunning the counting check on that detector.
Input-size and INT8 trade-offs follow, each with the same check.

### Across hosts (traced 2026-10-09)

The checks above compare runtimes on one host. Across CPU architectures, the same release
does not count identically. OpenCV's bilinear resize (the letterbox step) gives 8–15% of the
network-input pixels a value 1 grey level different (at most 2) on an Apple arm64 CPU than on
an x86 CPU. The detector amplifies this into score changes of up to about 0.18, enough to move
a few detections across the threshold. On the 36 clips of one kenai-channel day, the
development Mac counts 4 differently from the recorded Kaggle (x86) test evaluation.

The release image (linux/amd64), pulled by its manifest digest, reproduces the recorded test
evaluation's counts on the demo clip that differs. Development results measured on the Mac
can differ slightly on x86. The manual `host-parity` workflow checks a release image against
the public demo recordings stage by stage. Details and fingerprints are in
[demos](demos.md#results-depend-slightly-on-the-host).

## Releases, deployment verification and rollback (Stage 11)

**Release manifests.** Every release has an immutable manifest in `releases/manifests/`
(`scripts/make_release_manifest.py`, refused with uncommitted changes). It records:

- the code commit and the `uv.lock` hash;
- the dataset manifest and the training run (config hashes, training commit);
- the bundle's identity: pipeline config hash, checkpoint and ONNX hashes, input size and
  threshold, preprocessing version, tracker configuration, counting-policy and calibration
  versions;
- the evaluation it was selected and confirmed on, with report-file hashes and runtime parity
  checks;
- the image digest, which a release workflow fills in once.

A manifest can only be rewritten unchanged, apart from filling in the image digest once.

**Deployment verification.** `scripts/verify_deployment.py --url <service>` fails unless:

- `/health/ready` answers 200;
- `/v1/model-info` matches the served release's manifest field by field;
- an example job completes with that release.

Checked against a local deployment of `passagewatch-0.3.0`: it verified, and a manifest
altered in one field (the threshold) was reported as that exact difference.

**Rollback.** Point `bundles/active` back at the previous release (for example
`ln -sfn passagewatch-0.2.0 bundles/active`), restart the API and worker, and run
`verify_deployment.py`. With Docker Compose, the image must be the one built from that
release's code commit (recorded in its manifest); a bundle and an image from different
releases fail the start-up checks when their versions are incompatible. Jobs keep the
pipeline version recorded for them. `tests/integration/test_release.py` runs the procedure:
release A, then B, then a rollback to A. `/v1/model-info` matches each manifest in turn,
new jobs run the active release, and earlier jobs and their exports keep their own versions.

## Monitoring (Stage 11)

Docker Compose runs Prometheus (`prom/prometheus` v3.15.0, pinned by digest) on
`http://127.0.0.1:9090`, scraping two targets every 15 s.

| Target | Metrics |
|---|---|
| API `GET /metrics` (read from the database at scrape time) | `passagewatch_jobs{status}`, `passagewatch_queue_depth`, `passagewatch_oldest_queued_seconds`, `passagewatch_live_workers`, `passagewatch_review_events`, `passagewatch_release_info{pipeline_version, preprocessing_version, calibration_version, counting_policy, runtime}` |
| Worker, port 9100 inside the Compose network (`PASSAGEWATCH_WORKER_METRICS_PORT`) | `passagewatch_worker_jobs_total{outcome}`, `passagewatch_worker_job_seconds` (histogram), `passagewatch_worker_stage_seconds_total{stage}`, `passagewatch_worker_frames_total` |

The metrics carry counts and durations only: no recording content, identifiers or fish
counts.

**Alerts** (`deploy/prometheus/alerts.yml`, shown at `http://127.0.0.1:9090/alerts`; this
single-host deployment has no Alertmanager):

| Alert | Fires when | Severity |
|---|---|---|
| `PassageWatchApiDown` | Prometheus cannot scrape the API for 2 minutes | critical |
| `PassageWatchNoLiveWorker` | no worker of the active release heartbeats for 2 minutes | critical |
| `PassageWatchJobsFailing` | any job failed in the last 30 minutes | warning |
| `PassageWatchQueueBacklog` | a job has waited more than 30 minutes for a worker, for 5 minutes | warning |

CI checks the configuration with `promtool check config`, and tests each alert with
`promtool test rules` (`deploy/prometheus/alerts_test.yml`): every alert must fire when it
should and not before. The Compose smoke test requires both targets to be up in Prometheus
and the rules to be loaded.

**Publishing a release** (`.github/workflows/release.yml`). Pushing a tag
`passagewatch-<version>` runs:

1. the release gate (`scripts/check_release.py`): the release's manifest must exist, its code
   commit must be in the tagged history, it must record a development evaluation, and every
   runtime parity check must show 0 clips with different counts;
2. the fast tests;
3. an image build, pushed to `ghcr.io/<owner>/passagewatch:<version>`;
4. a GitHub Release carrying the image digest.

The release's inference bundle is then attached to it. `scripts/package_bundle.py --release
<version>` verifies `bundles/<version>/` against the manifest and writes a reproducible
`dist/<version>-bundle.tar.gz` (sorted members, fixed times, so the same bundle always gives
the same SHA-256) and its `.sha256`. The owner uploads both with `gh release upload`.
`scripts/fetch_bundle.py` downloads an archive and installs it only if it matches the
committed manifest: the pipeline configuration hash, the detector, preprocessing, tracker,
counting-policy and calibration fields, and the SHA-256 of each weight file. The archive
also carries `LICENSE` and `THIRD_PARTY_NOTICES.md` (CFC data: MIT; YOLOX and its COCO
weights: Apache-2.0).

The weights and data are not in git, so the evaluation and parity checks run locally before
tagging; the manifest is their record. The digest is then written into the manifest with
`scripts/record_image_digest.py` and committed by the project owner. No automation commits
to the repository.

Manifests must be generated from a commit that is already on `main`
(`make_release_manifest.py` refuses otherwise). Pull requests are rebase-merged, which gives
a branch's commits new hashes on `main`, so a commit recorded from a feature branch would not
exist in the released history.
