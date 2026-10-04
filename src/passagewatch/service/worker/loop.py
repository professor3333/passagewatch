"""The worker loop: lease a job, run it, publish the result; one job at a time.

- The pipeline (model) is loaded once, before the loop; the worker then reports itself ready
  and keeps reporting while idle, which ``/health/ready`` checks.
- Only jobs recorded with the worker's pipeline version are leased.
- While a job runs, a heartbeat thread (with its own database connection) extends the lease
  and records progress. If the lease is lost (for example, the job was recovered by another
  worker after a long stall), the next progress update aborts the job here without
  publishing anything.
- Failures are classified: bad media (``ValueError``, missing file) fails the job
  permanently; a lost lease just stops; anything else is retried while attempts remain.
- Artifacts are written before publication; a retried job rewrites them, and the result is
  published exactly once (see :class:`~passagewatch.service.jobs.JobStore`).
- Expired uploads are deleted periodically, unless a job still needs them.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import time

from passagewatch.service.artifacts import write_track_artifacts
from passagewatch.service.catalog import (
    expired_clips,
    get_clip,
    mark_clip_deleted,
    worker_heartbeat,
)
from passagewatch.service.db import connect
from passagewatch.service.jobs import QUEUED, RUNNING, Job, JobStore, LeaseLostError, utc_now
from passagewatch.service.metrics import WorkerMetrics
from passagewatch.service.settings import ServiceSettings
from passagewatch.service.worker.pipeline import InferencePipeline

logger = logging.getLogger(__name__)


class _JobHeartbeat(threading.Thread):
    """Extends a job's lease every ``interval`` seconds until stopped or the lease is lost."""

    def __init__(
        self, settings: ServiceSettings, job_id: str, worker_id: str, version: str
    ) -> None:
        super().__init__(daemon=True, name=f"heartbeat-{job_id}")
        self.settings = settings
        self.job_id = job_id
        self.worker_id = worker_id
        self.version = version
        self.progress = 0.0
        self.lost = threading.Event()
        self._stop_event = threading.Event()

    def run(self) -> None:
        conn = connect(self.settings.db_path)
        store = JobStore(conn)
        try:
            while not self._stop_event.wait(self.settings.heartbeat_seconds):
                now = utc_now()
                try:
                    store.heartbeat(
                        self.job_id,
                        self.worker_id,
                        lease_seconds=self.settings.lease_seconds,
                        progress=self.progress,
                        now=now,
                    )
                    worker_heartbeat(
                        conn, self.worker_id, self.version, now=now, current_job=self.job_id
                    )
                except LeaseLostError:
                    self.lost.set()
                    return
        finally:
            conn.close()

    def stop(self) -> None:
        self._stop_event.set()
        self.join()


def process_job(
    conn: sqlite3.Connection,
    settings: ServiceSettings,
    pipeline: InferencePipeline,
    job: Job,
    worker_id: str,
    metrics: WorkerMetrics | None = None,
) -> str:
    """Run one leased job to completion; returns its final status from this worker's view."""
    store = JobStore(conn)
    started = time.monotonic()
    heartbeat = _JobHeartbeat(settings, job.job_id, worker_id, pipeline.version)
    heartbeat.start()

    def progress(fraction: float) -> None:
        heartbeat.progress = fraction
        if heartbeat.lost.is_set():
            raise LeaseLostError(f"lease on {job.job_id} was lost")

    try:
        clip = get_clip(conn, job.clip_id)
        if clip is None or clip.deleted_at is not None:
            raise FileNotFoundError(f"clip {job.clip_id} is missing or deleted")
        result = pipeline.run(clip, settings.media_dir / clip.media_path, job.counting, progress)
        write_track_artifacts(
            settings.artifacts_dir / job.job_id,
            result.trajectories,
            result.counts,
            clip.framerate,
            result.review,
            pipeline.bundle.calibration_version,
        )
        heartbeat.stop()
        store.publish_result(
            job.job_id,
            worker_id,
            counts=result.summary(),
            tracks_artifact=job.job_id,
            now=utc_now(),
        )
        logger.info(
            "job succeeded",
            extra={
                "job_id": job.job_id,
                "worker_id": worker_id,
                **result.summary(),
                "frames": result.frames,
                "stage_seconds": result.stage_seconds,
            },
        )
        if metrics is not None:
            metrics.job_finished(
                "succeeded", time.monotonic() - started, result.stage_seconds, result.frames
            )
        return "succeeded"
    except LeaseLostError:
        logger.warning(
            "lease lost; stopping without publishing",
            extra={"job_id": job.job_id, "worker_id": worker_id},
        )
        return "lease_lost"
    except (ValueError, FileNotFoundError) as exc:
        return _fail(store, job, worker_id, str(exc), retryable=False)
    except Exception as exc:
        logger.exception("job failed", extra={"job_id": job.job_id, "worker_id": worker_id})
        return _fail(store, job, worker_id, f"{type(exc).__name__}: {exc}", retryable=True)
    finally:
        if heartbeat.is_alive():
            heartbeat.stop()


def _fail(store: JobStore, job: Job, worker_id: str, error: str, *, retryable: bool) -> str:
    try:
        updated = store.fail(job.job_id, worker_id, error=error, retryable=retryable, now=utc_now())
    except LeaseLostError:
        return "lease_lost"
    logger.warning(
        "job failed",
        extra={"job_id": job.job_id, "error": error, "status": updated.status},
    )
    return updated.status


def sweep_expired_uploads(conn: sqlite3.Connection, settings: ServiceSettings) -> list[str]:
    """Delete the media of expired uploads that no queued or running job needs."""
    removed = []
    for clip in expired_clips(conn, now=utc_now()):
        active = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE clip_id = ? AND status IN (?, ?)",
            (clip.clip_id, QUEUED, RUNNING),
        ).fetchone()[0]
        if active:
            continue
        shutil.rmtree(settings.media_dir / clip.clip_id, ignore_errors=True)
        mark_clip_deleted(conn, clip.clip_id, now=utc_now())
        removed.append(clip.clip_id)
    if removed:
        logger.info("deleted expired uploads", extra={"clips": len(removed)})
    return removed


def run_worker(
    settings: ServiceSettings,
    pipeline: InferencePipeline,
    worker_id: str,
    *,
    stop: threading.Event,
    max_jobs: int | None = None,
    metrics: WorkerMetrics | None = None,
) -> int:
    """Lease and run jobs until ``stop`` is set (or ``max_jobs`` have run); returns jobs run."""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings.db_path)
    store = JobStore(conn)
    done = 0
    last_sweep = 0.0
    try:
        worker_heartbeat(conn, worker_id, pipeline.version, now=utc_now())
        logger.info(
            "worker ready",
            extra={"worker_id": worker_id, "pipeline_version": pipeline.version},
        )
        while not stop.is_set() and (max_jobs is None or done < max_jobs):
            if time.monotonic() - last_sweep >= settings.retention_sweep_seconds:
                sweep_expired_uploads(conn, settings)
                last_sweep = time.monotonic()
            job = store.lease(
                worker_id,
                lease_seconds=settings.lease_seconds,
                now=utc_now(),
                pipeline_version=pipeline.version,
            )
            if job is None:
                worker_heartbeat(conn, worker_id, pipeline.version, now=utc_now())
                stop.wait(settings.worker_poll_seconds)
                continue
            logger.info(
                "job leased",
                extra={"job_id": job.job_id, "worker_id": worker_id, "attempt": job.attempts},
            )
            outcome = process_job(conn, settings, pipeline, job, worker_id, metrics)
            if metrics is not None and outcome != "succeeded":
                metrics.job_finished(outcome)
            worker_heartbeat(conn, worker_id, pipeline.version, now=utc_now())
            done += 1
    finally:
        conn.close()
    return done


def default_worker_id() -> str:
    import os
    import socket

    return f"{socket.gethostname()}-{os.getpid()}"
