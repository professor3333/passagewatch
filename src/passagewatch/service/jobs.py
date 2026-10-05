"""Durable analysis jobs: creation, leasing, heartbeats, retries, and result publication.

Lifecycle: ``queued`` → ``running`` (leased by one worker) → ``succeeded`` or ``failed``.

- **Idempotent creation.** A job request carries an idempotency key. Repeating the same
  request with the same key returns the existing job; reusing the key for a different
  request is an error.
- **Result cache.** The cache key is (recording SHA-256, pipeline config SHA-256, counting
  config). A request that matches a succeeded job is answered with a new job that is already
  succeeded, refers to the original (``cached_from``), and shares its result.
- **Leases.** :meth:`JobStore.lease` hands the oldest runnable job to one worker for
  ``lease_seconds``; the worker extends the lease with :meth:`JobStore.heartbeat`. A job
  whose lease expired (worker died) is leased again, up to ``max_attempts`` attempts in
  total; then it fails. A job running longer than its deadline fails.
- **Transactional publication.** :meth:`JobStore.publish_result` writes result revision 0 and
  marks the job succeeded in one transaction, and only for the current lease holder. A
  worker that lost its lease, or a retry of an already published job, cannot write a second
  result.

All methods take ``now`` explicitly, so lease and deadline behavior is testable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from passagewatch.service.db import transaction

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"


class IdempotencyConflictError(ValueError):
    """An idempotency key was reused for a different request."""


class QueueFullError(RuntimeError):
    """Too many jobs are queued or running."""


class LeaseLostError(RuntimeError):
    """The worker no longer holds the job's lease."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Job:
    job_id: str
    clip_id: str
    pipeline_version: str
    counting: dict[str, Any]
    status: str
    attempts: int
    max_attempts: int
    progress: float
    error: str | None
    cached_from: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    lease_owner: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(
            job_id=row["job_id"],
            clip_id=row["clip_id"],
            pipeline_version=row["pipeline_version"],
            counting=json.loads(row["counting_json"]),
            status=row["status"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            progress=row["progress"],
            error=row["error"],
            cached_from=row["cached_from"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            lease_owner=row["lease_owner"],
        )


@dataclass(frozen=True)
class CreatedJob:
    job: Job
    created: bool  # False when an identical request with the same key already existed


@dataclass(frozen=True)
class ResultRevision:
    job_id: str
    revision: int
    kind: str
    counts: dict[str, Any]
    tracks_artifact: str
    created_at: str
    decisions: dict[str, Any]


class JobStore:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        max_queue: int = 20,
        max_attempts: int = 3,
        deadline_seconds: int = 3600,
    ) -> None:
        self.conn = conn
        self.max_queue = max_queue
        self.max_attempts = max_attempts
        self.deadline_seconds = deadline_seconds

    # -- reads ---------------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else Job.from_row(row)

    def result(self, job_id: str, revision: int | None = None) -> ResultRevision | None:
        """The latest result revision of a job, or a given one."""
        query = "SELECT * FROM result_revisions WHERE job_id = ?"
        params: tuple[Any, ...] = (job_id,)
        if revision is not None:
            query += " AND revision = ?"
            params += (revision,)
        row = self.conn.execute(query + " ORDER BY revision DESC LIMIT 1", params).fetchone()
        if row is None:
            return None
        return ResultRevision(
            job_id=job_id,
            revision=row["revision"],
            kind=row["kind"],
            counts=json.loads(row["counts_json"]),
            tracks_artifact=row["tracks_artifact"],
            created_at=row["created_at"],
            decisions=json.loads(row["decisions_json"]),
        )

    # -- creation ------------------------------------------------------------------

    def create(
        self,
        *,
        clip_id: str,
        recording: dict[str, Any],
        pipeline_version: str,
        pipeline_config_sha256: str,
        counting: dict[str, Any],
        idempotency_key: str | None,
        now: datetime,
    ) -> CreatedJob:
        request = canonical_json(
            {"clip_id": clip_id, "pipeline_version": pipeline_version, "counting": counting}
        )
        request_sha256 = sha256_text(request)
        # The result cache: same recording (content and metadata), same pipeline
        # configuration, same counting configuration.
        cache_key = sha256_text(canonical_json([recording, pipeline_config_sha256, counting]))
        with transaction(self.conn):
            if idempotency_key is not None:
                row = self.conn.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if row is not None:
                    if row["request_sha256"] != request_sha256:
                        raise IdempotencyConflictError(
                            f"idempotency key {idempotency_key!r} was used for another request"
                        )
                    return CreatedJob(Job.from_row(row), created=False)

            job_id = new_id("job")
            cached = self.conn.execute(
                "SELECT job_id FROM jobs WHERE cache_key = ? AND status = ? AND cached_from IS NULL"
                " ORDER BY finished_at LIMIT 1",
                (cache_key, SUCCEEDED),
            ).fetchone()
            if cached is not None:
                self.conn.execute(
                    "INSERT INTO jobs (job_id, clip_id, pipeline_version, counting_json, cache_key,"
                    " idempotency_key, request_sha256, status, max_attempts, deadline_seconds,"
                    " progress, cached_from, created_at, finished_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?, ?, ?)",
                    (
                        job_id,
                        clip_id,
                        pipeline_version,
                        canonical_json(counting),
                        cache_key,
                        idempotency_key,
                        request_sha256,
                        SUCCEEDED,
                        self.max_attempts,
                        self.deadline_seconds,
                        cached["job_id"],
                        iso(now),
                        iso(now),
                    ),
                )
                # The cached job gets its own copy of the automatic result, so its reviews
                # (later revisions) are its own and never mix with the original's.
                self.conn.execute(
                    "INSERT INTO result_revisions (job_id, revision, kind, created_at, counts_json,"
                    " tracks_artifact, decisions_json) SELECT ?, 0, 'automatic', ?, counts_json,"
                    " tracks_artifact, decisions_json FROM result_revisions"
                    " WHERE job_id = ? AND revision = 0",
                    (job_id, iso(now), cached["job_id"]),
                )
            else:
                active = self.conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN (?, ?)", (QUEUED, RUNNING)
                ).fetchone()[0]
                if active >= self.max_queue:
                    raise QueueFullError(
                        f"{active} jobs are queued or running (limit {self.max_queue})"
                    )
                self.conn.execute(
                    "INSERT INTO jobs (job_id, clip_id, pipeline_version, counting_json, cache_key,"
                    " idempotency_key, request_sha256, status, max_attempts, deadline_seconds,"
                    " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        job_id,
                        clip_id,
                        pipeline_version,
                        canonical_json(counting),
                        cache_key,
                        idempotency_key,
                        request_sha256,
                        QUEUED,
                        self.max_attempts,
                        self.deadline_seconds,
                        iso(now),
                    ),
                )
        job = self.get(job_id)
        assert job is not None
        return CreatedJob(job, created=True)

    # -- worker side ---------------------------------------------------------------

    def lease(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        now: datetime,
        pipeline_version: str | None = None,
    ) -> Job | None:
        """Lease the oldest runnable job: queued, or running with an expired lease.

        With ``pipeline_version``, only jobs recorded with that version are leased: a worker
        never runs a job with a different model than the job was created for.
        """
        with transaction(self.conn):
            self._expire(now)
            version_clause = "" if pipeline_version is None else " AND pipeline_version = ?"
            params: tuple[Any, ...] = (QUEUED, RUNNING, iso(now))
            if pipeline_version is not None:
                params += (pipeline_version,)
            row = self.conn.execute(
                "SELECT * FROM jobs WHERE (status = ? OR (status = ? AND lease_expires_at < ?))"
                + version_clause
                + " ORDER BY created_at LIMIT 1",
                params,
            ).fetchone()
            if row is None:
                return None
            if row["attempts"] >= row["max_attempts"]:
                self._finish(row["job_id"], FAILED, "lease expired after the last attempt", now)
                return None
            self.conn.execute(
                "UPDATE jobs SET status = ?, attempts = attempts + 1, lease_owner = ?,"
                " lease_expires_at = ?, heartbeat_at = ?, started_at = COALESCE(started_at, ?),"
                " error = NULL WHERE job_id = ?",
                (
                    RUNNING,
                    worker_id,
                    iso(now + timedelta(seconds=lease_seconds)),
                    iso(now),
                    iso(now),
                    row["job_id"],
                ),
            )
        return self.get(row["job_id"])

    def heartbeat(
        self, job_id: str, worker_id: str, *, lease_seconds: int, progress: float, now: datetime
    ) -> None:
        """Extend the lease; raises :class:`LeaseLostError` if the worker no longer holds it."""
        with transaction(self.conn):
            updated = self.conn.execute(
                "UPDATE jobs SET lease_expires_at = ?, heartbeat_at = ?, progress = ?"
                " WHERE job_id = ? AND status = ? AND lease_owner = ? AND lease_expires_at >= ?",
                (
                    iso(now + timedelta(seconds=lease_seconds)),
                    iso(now),
                    progress,
                    job_id,
                    RUNNING,
                    worker_id,
                    iso(now),
                ),
            ).rowcount
        if updated != 1:
            raise LeaseLostError(f"{worker_id} no longer holds the lease on {job_id}")

    def publish_result(
        self,
        job_id: str,
        worker_id: str,
        *,
        counts: dict[str, Any],
        tracks_artifact: str,
        now: datetime,
    ) -> None:
        """Atomically store result revision 0 and mark the job succeeded."""
        with transaction(self.conn):
            row = self.conn.execute(
                "SELECT status, lease_owner, lease_expires_at FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != RUNNING
                or row["lease_owner"] != worker_id
                or row["lease_expires_at"] < iso(now)
            ):
                raise LeaseLostError(f"{worker_id} cannot publish {job_id}: lease not held")
            self.conn.execute(
                "INSERT INTO result_revisions (job_id, revision, kind, created_at, counts_json,"
                " tracks_artifact) VALUES (?, 0, 'automatic', ?, ?, ?)",
                (job_id, iso(now), canonical_json(counts), tracks_artifact),
            )
            self._finish(job_id, SUCCEEDED, None, now)

    def fail(
        self, job_id: str, worker_id: str, *, error: str, retryable: bool, now: datetime
    ) -> Job:
        """Record a failure; a retryable one goes back to the queue while attempts remain."""
        with transaction(self.conn):
            row = self.conn.execute(
                "SELECT attempts, max_attempts, lease_owner, status FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None or row["status"] != RUNNING or row["lease_owner"] != worker_id:
                raise LeaseLostError(f"{worker_id} cannot fail {job_id}: lease not held")
            if retryable and row["attempts"] < row["max_attempts"]:
                self.conn.execute(
                    "UPDATE jobs SET status = ?, lease_owner = NULL, lease_expires_at = NULL,"
                    " error = ? WHERE job_id = ?",
                    (QUEUED, error, job_id),
                )
            else:
                self._finish(job_id, FAILED, error, now)
        job = self.get(job_id)
        assert job is not None
        return job

    # -- internals -----------------------------------------------------------------

    def _expire(self, now: datetime) -> None:
        """Fail running jobs that have exceeded their deadline."""
        for row in self.conn.execute(
            "SELECT job_id, started_at, deadline_seconds FROM jobs WHERE status = ?", (RUNNING,)
        ).fetchall():
            started = datetime.fromisoformat(row["started_at"])
            if now - started > timedelta(seconds=row["deadline_seconds"]):
                self._finish(row["job_id"], FAILED, "deadline exceeded", now)

    def _finish(self, job_id: str, status: str, error: str | None, now: datetime) -> None:
        self.conn.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ?, lease_owner = NULL,"
            " lease_expires_at = NULL, progress = CASE WHEN ? = 'succeeded' THEN 1.0"
            " ELSE progress END WHERE job_id = ?",
            (status, error, iso(now), status, job_id),
        )
