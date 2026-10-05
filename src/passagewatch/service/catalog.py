"""Clips and pipeline versions: the records that jobs refer to.

A pipeline version names one complete inference configuration (detector checkpoint,
preprocessing, tracker, counting policy). Versions are immutable: registering an existing
version name with a different configuration is refused. Every job records the version it
ran with, so existing jobs keep their versions when a new release is activated.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from passagewatch.service.db import transaction
from passagewatch.service.jobs import canonical_json, iso, new_id, sha256_text


class PipelineVersionConflictError(ValueError):
    """A pipeline version name is already registered with a different configuration."""


@dataclass(frozen=True)
class PipelineVersion:
    pipeline_version: str
    config: dict[str, Any]
    config_sha256: str


def register_pipeline_version(
    conn: sqlite3.Connection, version: str, config: dict[str, Any], *, now: datetime
) -> PipelineVersion:
    digest = sha256_text(canonical_json(config))
    with transaction(conn):
        row = conn.execute(
            "SELECT config_sha256 FROM pipeline_versions WHERE pipeline_version = ?", (version,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO pipeline_versions (pipeline_version, created_at, config_sha256,"
                " config_json) VALUES (?, ?, ?, ?)",
                (version, iso(now), digest, canonical_json(config)),
            )
        elif row["config_sha256"] != digest:
            raise PipelineVersionConflictError(
                f"pipeline version {version!r} is already registered with another configuration"
            )
    return PipelineVersion(version, config, digest)


def get_pipeline_version(conn: sqlite3.Connection, version: str) -> PipelineVersion | None:
    row = conn.execute(
        "SELECT * FROM pipeline_versions WHERE pipeline_version = ?", (version,)
    ).fetchone()
    if row is None:
        return None
    return PipelineVersion(version, json.loads(row["config_json"]), row["config_sha256"])


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    source: str
    sha256: str
    media_kind: str
    media_path: str
    num_frames: int
    width: int
    height: int
    framerate: float
    x_meter_start: float
    x_meter_stop: float
    y_meter_start: float
    y_meter_stop: float
    created_at: str
    expires_at: str | None
    deleted_at: str | None

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.framerate

    def identity(self) -> dict[str, Any]:
        """Everything about the recording that the analysis depends on, for the result cache.

        The same bytes with another frame rate or sonar window give other tracks (motion
        in meters, gaps in seconds) and other timestamps, so they are another recording.
        """
        return {
            "sha256": self.sha256,
            "media_kind": self.media_kind,
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "framerate": self.framerate,
            "meters": [
                self.x_meter_start,
                self.x_meter_stop,
                self.y_meter_start,
                self.y_meter_stop,
            ],
        }


_CLIP_FIELDS = (
    "clip_id",
    "source",
    "sha256",
    "media_kind",
    "media_path",
    "num_frames",
    "width",
    "height",
    "framerate",
    "x_meter_start",
    "x_meter_stop",
    "y_meter_start",
    "y_meter_stop",
    "created_at",
    "expires_at",
    "deleted_at",
)


def add_clip(
    conn: sqlite3.Connection,
    *,
    source: str,
    sha256: str,
    media_kind: str,
    media_path: str,
    num_frames: int,
    width: int,
    height: int,
    framerate: float,
    meters: tuple[float, float, float, float],
    now: datetime,
    expires_at: datetime | None,
    clip_id: str | None = None,
) -> ClipRecord:
    record = ClipRecord(
        clip_id=clip_id or new_id("clip"),
        source=source,
        sha256=sha256,
        media_kind=media_kind,
        media_path=media_path,
        num_frames=num_frames,
        width=width,
        height=height,
        framerate=framerate,
        x_meter_start=meters[0],
        x_meter_stop=meters[1],
        y_meter_start=meters[2],
        y_meter_stop=meters[3],
        created_at=iso(now),
        expires_at=None if expires_at is None else iso(expires_at),
        deleted_at=None,
    )
    with transaction(conn):
        conn.execute(
            f"INSERT INTO clips ({', '.join(_CLIP_FIELDS)})"
            f" VALUES ({', '.join('?' for _ in _CLIP_FIELDS)})",
            tuple(getattr(record, f) for f in _CLIP_FIELDS),
        )
    return record


def get_clip(conn: sqlite3.Connection, clip_id: str) -> ClipRecord | None:
    row = conn.execute("SELECT * FROM clips WHERE clip_id = ?", (clip_id,)).fetchone()
    return None if row is None else ClipRecord(**{f: row[f] for f in _CLIP_FIELDS})


def mark_clip_deleted(conn: sqlite3.Connection, clip_id: str, *, now: datetime) -> bool:
    """Record a clip's media as deleted (the row stays: jobs and reports refer to it)."""
    with transaction(conn):
        updated = conn.execute(
            "UPDATE clips SET deleted_at = ? WHERE clip_id = ? AND deleted_at IS NULL",
            (iso(now), clip_id),
        ).rowcount
    return updated == 1


def expired_clips(conn: sqlite3.Connection, *, now: datetime) -> list[ClipRecord]:
    rows = conn.execute(
        "SELECT * FROM clips WHERE deleted_at IS NULL AND expires_at IS NOT NULL"
        " AND expires_at <= ?",
        (iso(now),),
    ).fetchall()
    return [ClipRecord(**{f: r[f] for f in _CLIP_FIELDS}) for r in rows]


@dataclass(frozen=True)
class WorkerRecord:
    worker_id: str
    pipeline_version: str
    ready_at: str
    heartbeat_at: str
    current_job: str | None


def worker_heartbeat(
    conn: sqlite3.Connection,
    worker_id: str,
    pipeline_version: str,
    *,
    now: datetime,
    current_job: str | None = None,
) -> None:
    """Record that a worker has its model loaded and is alive (first call marks it ready)."""
    with transaction(conn):
        conn.execute(
            "INSERT INTO workers (worker_id, pipeline_version, ready_at, heartbeat_at, current_job)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT (worker_id) DO UPDATE SET"
            " pipeline_version = excluded.pipeline_version,"
            " heartbeat_at = excluded.heartbeat_at, current_job = excluded.current_job",
            (worker_id, pipeline_version, iso(now), iso(now), current_job),
        )


def live_workers(
    conn: sqlite3.Connection, pipeline_version: str, *, now: datetime, stale_seconds: float
) -> list[WorkerRecord]:
    cutoff = iso(now - timedelta(seconds=stale_seconds))
    rows = conn.execute(
        "SELECT * FROM workers WHERE pipeline_version = ? AND heartbeat_at >= ?",
        (pipeline_version, cutoff),
    ).fetchall()
    fields = ("worker_id", "pipeline_version", "ready_at", "heartbeat_at", "current_job")
    return [WorkerRecord(**{f: r[f] for f in fields}) for r in rows]
