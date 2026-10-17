# Service Architecture

PassageWatch runs as two processes from one Docker image, sharing one data volume, with
Prometheus beside them. The API never runs inference: it records jobs and serves results. A
separate worker leases jobs, runs the release bundle's pipeline, and publishes results.

```mermaid
flowchart LR
    U[Technician / client] -->|HTTP| A[API: FastAPI<br/>uvicorn]
    A -->|uploads| M[(media/)]
    A <-->|jobs, results, revisions| D[(SQLite, WAL)]
    W[Worker: python -m<br/>passagewatch.service.worker] <-->|lease, heartbeat, publish| D
    M --> W
    B[(bundles/active<br/>read-only)] --> W
    B -->|bundle.json only| A
    W -->|tracks.parquet,<br/>observations.parquet| R[(artifacts/)]
    R --> A
    P[Prometheus] -->|/metrics| A
    P -->|:9100| W
```

| Component | Code | Role |
|---|---|---|
| API | `passagewatch.service.api` | Uploads, job creation (202), status, results, tracks, model info, health |
| Worker | `passagewatch.service.worker` | Loads the bundle once; leases and runs one job at a time |
| Job store | `passagewatch.service.jobs` | Durable job lifecycle in SQLite: leases, retries, deadlines, publication |
| Catalog | `passagewatch.service.catalog` | Clips (with retention), pipeline versions, worker readiness |
| Media | `passagewatch.service.media` | Streamed uploads with a size limit, validation, bounded-memory frame reading |
| Bundles | `passagewatch.service.bundle` | Immutable release bundles and the active-release pointer |
| Artifacts | `passagewatch.service.artifacts` | Per-job Parquet files with trajectories and their boxes |
| Reviews and export | `passagewatch.service.reviews`, `passagewatch.service.export` | Append-only review events, result revisions, JSON/CSV reports |
| Demos | `passagewatch.service.demos` | The three demo examples and their cached results ([demos](demos.md)) |
| Metrics | `passagewatch.service.metrics` | Prometheus metrics: the API's from the database, the worker's on port 9100 |
| Releases | `passagewatch.service.release`, `passagewatch.service.bundle_archive` | Release manifests, deployment checks, bundle archives on GitHub releases |
| Review interface | `frontend/` (TypeScript, Vite) | Upload, progress, playback with overlays, review actions, export; built into the image and served by the API at `/` |

## Analysis flow

1. `POST /v1/clips` streams the recording to `media/<clip_id>/` while hashing it (SHA-256),
   validates it (a ZIP of frames `0..N-1` or a decodable video), and records it, with its
   frame rate and the sonar window in meters, and an expiry time.
2. `POST /v1/jobs` records a job against the **active pipeline version** and returns
   `202 Accepted` at once, with the status URL in `Location`. An `Idempotency-Key` makes the
   request safe to retry. An identical analysis is answered from the result cache:
   the same recording (its SHA-256 **and** its frame count, size, frame rate and sonar
   window, which change tracking and timestamps), the same pipeline configuration and the
   same counting configuration.
3. The worker leases the oldest queued job of **its own** pipeline version. It decodes
   frames in batches and keeps only their detections, then tracks sequentially, then counts
   completed trajectories with the job's counting line (`cfc-compatible-v1`).
4. The worker writes the artifacts, then publishes result revision 0 and marks the job
   succeeded, in one transaction.
5. `GET /v1/jobs/{id}`, `/results` and `/tracks` serve status, counts and trajectories.
   Results keep automatic and reviewed counts in separate fields.

## Review and export

Revision 0 of a result is the automatic one and never changes. A reviewer submits
corrections with `POST /v1/jobs/{id}/reviews`. Each correction names the revision it was
based on, is stored as an append-only `review_events` row, and produces a new
`result_revisions` row with the decisions so far and the reviewed counts. If the base
revision is not the latest, because another reviewer saved first, the correction is refused
with **409 Conflict**. Automatic and reviewed counts are always separate fields.

| Action | Effect on the reviewed counts |
|---|---|
| `accept` (track) | Counts with its automatic direction (confirmed) |
| `reject` (track or added passage) | Does not count (not a fish, a duplicate fragment, a mistaken addition) |
| `set_direction` (track) | Counts with the reviewer's direction |
| `mark_unresolved` (track) | Excluded from the counts and listed as unresolved |
| `add_passage` (direction, frame) | A fish the model missed; counts, with that frame as evidence (optionally tied to an `audit_window`) |
| `mark_audited` (`audit_window`) | None; records that a random audit window was watched |

`GET /v1/jobs/{id}/export?format=json|csv&revision=N` produces a report for any revision. It
contains:
- the recording ID and SHA-256;
- the counting policy, line and orientation;
- the automatic and reviewed counts (upstream/downstream when an orientation is set);
- the unresolved cases;
- the pipeline version, config hash and detector checkpoint hash;
- every track and added passage with its evidence frames and times, review state and final
  direction, and (with a calibration version) its triage state and review score;
- the random audit windows, which were checked, and the passages found in them;
- the review history up to that revision.

The CSV has the metadata and counts as `# key,value` lines above one row per case.

For the review interface:
- `GET /v1/clips/{id}` returns the clip metadata;
- `GET /v1/clips/{id}/frames/{n}` returns a frame as an image;
- `GET /v1/jobs/{id}/observations?start=&stop=` returns every tracked box in a window of up
  to 500 frames, for overlays;
- `/tracks` also returns each track's review state and final direction, and its triage,
  review score and reasons; `?order=queue` lists tracks in review-queue order;
- `GET /v1/jobs/{id}/audit` returns the job's random audit windows and their state.
- `GET /v1/demos` lists the demo examples, with each one's clip and precomputed job once
  the active release has analysed it ([demos](demos.md)). A demo's recording never expires,
  and deleting it is refused with `403`.

Triage, review scores and audit windows exist when the release declares a calibration
version (`review-v0`); they are computed by the worker and stored with the predictions. See
[review.md](review.md).

## Job lifecycle

```mermaid
stateDiagram-v2
    [*] --> queued: POST /v1/jobs
    queued --> running: worker leases
    running --> running: heartbeat extends lease
    running --> succeeded: result published (once, by lease holder)
    running --> queued: retryable error, attempts left
    running --> failed: bad media, deadline, or attempts exhausted
    running --> running: lease expired → another worker leases (attempt + 1)
```

| Guarantee | How |
|---|---|
| No request waits for inference | Jobs are asynchronous; the API returns 202 |
| One worker per job | Leases are taken in `BEGIN IMMEDIATE` transactions (tested with racing connections) |
| Crashed workers do not lose jobs | An expired lease is leased again, up to `max_attempts`; tested with a SIGKILLed worker process |
| No duplicate results | Revision 0 and the success status are written together, only by the current lease holder |
| A stalled worker cannot publish late | Its heartbeat fails once the lease is taken over, and it stops at the next batch |
| No silent model switch | Workers lease only jobs of their own pipeline version; a mismatched bundle refuses to load |
| History is not rewritten | `result_revisions` and `review_events` are append-only (database triggers) |

## Storage

| Path (inside `/data`) | Contents |
|---|---|
| `service.db` | SQLite (WAL, schema v3): `clips`, `jobs`, `pipeline_versions`, `result_revisions` (with each revision's review decisions), `review_events`, `workers` |
| `media/<clip_id>/` | Uploaded recording; deleted after its retention period unless a job needs it |
| `artifacts/<job_id>/` | `tracks.parquet` (one row per trajectory), `observations.parquet` (every box), `audit.json` (audit windows, with a calibration version) |

## Release bundles

A bundle (`bundles/<version>/`) holds `bundle.json`, the detector checkpoint
(`detector.pt`) and, for an ONNX release, its ONNX export (`detector.onnx`). It is built by
`scripts/build_bundle.py` from a training checkpoint, and records:
- the checkpoint's SHA-256, and the ONNX file's SHA-256 and the runtime (`onnxruntime`);
- the detector's size, input size, training run and epoch, and the score threshold;
- the preprocessing version;
- the tracker configuration;
- the counting policy and the calibration version;
- provenance: the training manifest hash and where the selection is documented.

Versions are immutable. `bundles/active` is a symlink to the active version. The API reads
only `bundle.json`, to register the pipeline version and answer `/v1/model-info`. The
worker loads the weights, after checking their SHA-256 and the preprocessing version.
Bundles are mounted read-only into the containers and never committed or built into the
image. Each release's bundle is attached to its GitHub release as a reproducible archive;
`scripts/fetch_bundle.py` downloads it and checks it against the release manifest in
`releases/manifests/`.

Current release: `passagewatch-0.3.2` (the pipeline of 0.3.0 and 0.3.1). It is YOLOX-Tiny `yolox-tiny-t1` (temporal input,
`letterbox-temporal3-v1`) epoch 25, with score threshold 0.4, run by ONNX Runtime on CPU,
the `classical-v2` tracker and calibration version `review-v0`, as selected in
[error_analysis.md](error_analysis.md) and evaluated in [test_results.md](test_results.md).
A regression test checks that the service counts a real kenai-val clip exactly as the
offline evaluation did.

## Configuration

All settings are `PASSAGEWATCH_*` environment variables (`passagewatch.service.settings`):

| Variable | Default | Meaning |
|---|---|---|
| `PASSAGEWATCH_DATA_DIR` | `var` (`/data` in the image) | Database, uploads, artifacts |
| `PASSAGEWATCH_BUNDLE_DIR` | `bundles/active` (`/bundles/active`) | Active release bundle |
| `PASSAGEWATCH_FRONTEND_DIR` | `frontend/dist` (`/app/frontend/dist`) | Built review interface, served at `/` if present |
| `PASSAGEWATCH_MAX_UPLOAD_BYTES` | 524288000 (500 MiB) | Upload size limit (413 above) |
| `PASSAGEWATCH_MAX_FRAMES` | 6000 | Frame limit per recording |
| `PASSAGEWATCH_UPLOAD_RETENTION_HOURS` | 24 | Uploads are deleted after this |
| `PASSAGEWATCH_MAX_QUEUE` | 20 | Queued + running jobs before 503 |
| `PASSAGEWATCH_MAX_ATTEMPTS` | 3 | Attempts per job |
| `PASSAGEWATCH_JOB_DEADLINE_SECONDS` | 3600 | A job running longer fails |
| `PASSAGEWATCH_LEASE_SECONDS` | 120 | Lease length; extended by heartbeats |
| `PASSAGEWATCH_HEARTBEAT_SECONDS` | 20 | Worker heartbeat interval |
| `PASSAGEWATCH_TRACKS_PAGE_LIMIT` | 200 | Maximum `limit` for `/tracks` |
| `PASSAGEWATCH_DEVICE` | `cpu` | Inference device (`auto` tries CUDA, then MPS) |
| `PASSAGEWATCH_INFERENCE_BATCH` | 8 | Frames per detector batch |
| `PASSAGEWATCH_TORCH_THREADS` | unset | CPU threads for detection; set it to the host's performance cores ([operations](operations.md)) |
| `PASSAGEWATCH_WORKER_METRICS_PORT` | unset (`9100` in Compose) | Port of the worker's Prometheus metrics; unset disables them |
| `PASSAGEWATCH_WORKER_POLL_SECONDS` | 2 | Idle polling interval |
| `PASSAGEWATCH_RETENTION_SWEEP_SECONDS` | 600 | How often expired uploads are deleted |
| `PASSAGEWATCH_WORKER_STALE_SECONDS` | 60 | A worker silent this long no longer counts for readiness |
| `PASSAGEWATCH_DEMO_CATALOG` | unset (`/app/demos/catalog.json` in the image) | Demo catalog ([demos](demos.md)); unset means no demos |
| `PASSAGEWATCH_STUDY_MODE`, `PASSAGEWATCH_STUDY_PLAN` | off, unset | Study pages for the usability study ([usability_study.md](usability_study.md)) |

## Logs

Both processes write one JSON object per line to stdout (`passagewatch.service.logs`),
including Uvicorn's access log. Worker events carry `job_id`, `worker_id`, `attempt` and
result fields as top-level keys.

## Known limits

- **CPU inference takes about 0.1 s per frame.** The release pipeline runs at about 98.5
  ms per frame (about 10 frames/s) on the development M1 CPU with 4 threads; the detector is
  about 97% of the time. INT8 and smaller input sizes were not tried
  ([operations](operations.md)). The image is about 2 GB.
- **Counts can differ slightly between CPU architectures**, because OpenCV's resize rounds a
  few pixels differently on arm64 and x86 ([demos](demos.md#results-depend-slightly-on-the-host)).
- **One host, one worker.** SQLite and local volumes are deliberate for this scale (see
  [design](design.md) §17).
- **Not built:** a reverse proxy with HTTPS in front of the API, needed for a public
  deployment. The Compose file publishes the API on 127.0.0.1 only.
