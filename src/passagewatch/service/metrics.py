"""Prometheus metrics for the API (read from the database at scrape time) and the worker.

API, ``GET /metrics``:

- ``passagewatch_jobs{status}``: jobs by status (queued, running, succeeded, failed);
- ``passagewatch_queue_depth`` and ``passagewatch_oldest_queued_seconds``;
- ``passagewatch_live_workers``: workers of the active release heartbeating recently;
- ``passagewatch_review_events``: review events stored;
- ``passagewatch_release_info{pipeline_version, preprocessing_version, calibration_version,
  counting_policy, runtime}``: always 1, for the active release.

Worker, on its own port (``PASSAGEWATCH_WORKER_METRICS_PORT``):

- ``passagewatch_worker_jobs_total{outcome}``: jobs finished, by outcome;
- ``passagewatch_worker_job_seconds``: wall time of successful jobs (histogram);
- ``passagewatch_worker_stage_seconds_total{stage}``: time per pipeline stage;
- ``passagewatch_worker_frames_total``: frames processed.

Values are counts and durations only: no recording content, IDs or counts of fish.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily
from prometheus_client.registry import Collector

from passagewatch.service.bundle import ReleaseBundle
from passagewatch.service.jobs import iso

JOB_STATUSES = ("queued", "running", "succeeded", "failed")


class ServiceCollector(Collector):
    """Reads job, queue, worker and review state from the database on every scrape."""

    def __init__(
        self,
        connect: Callable[[], sqlite3.Connection],
        bundle: Callable[[], ReleaseBundle | None],
        now: Callable[[], datetime],
        stale_seconds: float,
    ) -> None:
        self.connect = connect
        self.bundle = bundle
        self.now = now
        self.stale_seconds = stale_seconds

    def collect(self) -> Iterator[GaugeMetricFamily | InfoMetricFamily]:
        conn = self.connect()
        try:
            now = self.now()
            counts = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status"))
            jobs = GaugeMetricFamily("passagewatch_jobs", "Jobs by status", labels=["status"])
            for status in JOB_STATUSES:
                jobs.add_metric([status], counts.get(status, 0))
            yield jobs
            yield GaugeMetricFamily(
                "passagewatch_queue_depth",
                "Jobs waiting for a worker",
                value=counts.get("queued", 0),
            )
            oldest = conn.execute(
                "SELECT MIN(created_at) FROM jobs WHERE status = 'queued'"
            ).fetchone()[0]
            age = 0.0 if oldest is None else (now - datetime.fromisoformat(oldest)).total_seconds()
            yield GaugeMetricFamily(
                "passagewatch_oldest_queued_seconds", "Age of the oldest queued job", value=age
            )
            bundle = self.bundle()
            live = 0
            if bundle is not None:
                cutoff = iso(now - timedelta(seconds=self.stale_seconds))
                live = conn.execute(
                    "SELECT COUNT(*) FROM workers WHERE pipeline_version = ? AND heartbeat_at >= ?",
                    (bundle.pipeline_version, cutoff),
                ).fetchone()[0]
            yield GaugeMetricFamily(
                "passagewatch_live_workers", "Live workers of the active release", value=live
            )
            events = conn.execute("SELECT COUNT(*) FROM review_events").fetchone()[0]
            yield GaugeMetricFamily(
                "passagewatch_review_events", "Review events stored", value=events
            )
            if bundle is not None:
                info = InfoMetricFamily("passagewatch_release", "The active release")
                info.add_metric(
                    [],
                    {
                        "pipeline_version": bundle.pipeline_version,
                        "preprocessing_version": bundle.preprocessing_version,
                        "calibration_version": bundle.calibration_version or "none",
                        "counting_policy": bundle.counting_policy,
                        "runtime": bundle.detector.runtime,
                    },
                )
                yield info
        finally:
            conn.close()


class WorkerMetrics:
    """Counters and histograms the worker updates as it finishes jobs."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.jobs = Counter(
            "passagewatch_worker_jobs_total",
            "Jobs finished by this worker, by outcome",
            ["outcome"],
            registry=self.registry,
        )
        self.job_seconds = Histogram(
            "passagewatch_worker_job_seconds",
            "Wall time of successful jobs",
            buckets=(5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600),
            registry=self.registry,
        )
        self.stage_seconds = Counter(
            "passagewatch_worker_stage_seconds_total",
            "Time spent per pipeline stage",
            ["stage"],
            registry=self.registry,
        )
        self.frames = Counter(
            "passagewatch_worker_frames_total", "Frames processed", registry=self.registry
        )

    def job_finished(
        self,
        outcome: str,
        seconds: float | None = None,
        stage_seconds: dict[str, float] | None = None,
        frames: int = 0,
    ) -> None:
        self.jobs.labels(outcome=outcome).inc()
        if seconds is not None and outcome == "succeeded":
            self.job_seconds.observe(seconds)
        for stage, value in (stage_seconds or {}).items():
            self.stage_seconds.labels(stage=stage).inc(value)
        if frames:
            self.frames.inc(frames)
