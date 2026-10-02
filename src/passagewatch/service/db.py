"""SQLite persistence for the service: connection settings and the versioned schema.

SQLite runs in WAL mode so the API can read while the worker writes. Every state change
that must be atomic runs inside :func:`transaction` (``BEGIN IMMEDIATE``), which takes the
write lock up front, so two processes can never both lease the same job or publish the same
result. Timestamps are ISO-8601 UTC strings, which sort in time order.

Original predictions are immutable: ``result_revisions`` and ``review_events`` are only ever
inserted into, never updated. Dense per-frame trajectories live in artifact files, not in
rows.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 3

# Version 2: workers report readiness (model loaded) and liveness for /health/ready.
_SCHEMA_V2 = """
CREATE TABLE workers (
    worker_id         TEXT PRIMARY KEY,
    pipeline_version  TEXT NOT NULL,
    ready_at          TEXT NOT NULL,
    heartbeat_at      TEXT NOT NULL,
    current_job       TEXT
);
"""

# Version 3: each result revision stores its review decisions (per track and added passages).
_SCHEMA_V3 = """
ALTER TABLE result_revisions ADD COLUMN decisions_json TEXT NOT NULL DEFAULT '{}';
"""

_SCHEMA_V1 = """
CREATE TABLE clips (
    clip_id        TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    source         TEXT NOT NULL CHECK (source IN ('upload', 'example')),
    sha256         TEXT NOT NULL,
    media_kind     TEXT NOT NULL CHECK (media_kind IN ('video', 'frames')),
    media_path     TEXT NOT NULL,
    num_frames     INTEGER NOT NULL CHECK (num_frames > 0),
    width          INTEGER NOT NULL CHECK (width > 0),
    height         INTEGER NOT NULL CHECK (height > 0),
    framerate      REAL NOT NULL CHECK (framerate > 0),
    x_meter_start  REAL NOT NULL,
    x_meter_stop   REAL NOT NULL,
    y_meter_start  REAL NOT NULL,
    y_meter_stop   REAL NOT NULL,
    expires_at     TEXT,
    deleted_at     TEXT
);

CREATE TABLE pipeline_versions (
    pipeline_version  TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    config_sha256     TEXT NOT NULL,
    config_json       TEXT NOT NULL
);

CREATE TABLE jobs (
    job_id            TEXT PRIMARY KEY,
    clip_id           TEXT NOT NULL REFERENCES clips (clip_id),
    pipeline_version  TEXT NOT NULL REFERENCES pipeline_versions (pipeline_version),
    counting_json     TEXT NOT NULL,
    cache_key         TEXT NOT NULL,
    idempotency_key   TEXT UNIQUE,
    request_sha256    TEXT NOT NULL,
    status            TEXT NOT NULL
                      CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    attempts          INTEGER NOT NULL DEFAULT 0,
    max_attempts      INTEGER NOT NULL CHECK (max_attempts > 0),
    deadline_seconds  INTEGER NOT NULL CHECK (deadline_seconds > 0),
    lease_owner       TEXT,
    lease_expires_at  TEXT,
    heartbeat_at      TEXT,
    progress          REAL NOT NULL DEFAULT 0,
    error             TEXT,
    cached_from       TEXT REFERENCES jobs (job_id),
    created_at        TEXT NOT NULL,
    started_at        TEXT,
    finished_at       TEXT
);
CREATE INDEX jobs_by_status ON jobs (status, created_at);
CREATE INDEX jobs_by_cache_key ON jobs (cache_key, status);

CREATE TABLE result_revisions (
    job_id           TEXT NOT NULL REFERENCES jobs (job_id),
    revision         INTEGER NOT NULL CHECK (revision >= 0),
    kind             TEXT NOT NULL CHECK (kind IN ('automatic', 'reviewed')),
    created_at       TEXT NOT NULL,
    counts_json      TEXT NOT NULL,
    tracks_artifact  TEXT NOT NULL,
    review_event_id  INTEGER REFERENCES review_events (event_id),
    PRIMARY KEY (job_id, revision)
);

CREATE TABLE review_events (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id              TEXT NOT NULL REFERENCES jobs (job_id),
    base_revision       INTEGER NOT NULL,
    resulting_revision  INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    payload_json        TEXT NOT NULL
);

-- Original predictions and review history are append-only.
CREATE TRIGGER result_revisions_immutable BEFORE UPDATE ON result_revisions
BEGIN SELECT RAISE(ABORT, 'result revisions are immutable'); END;
CREATE TRIGGER result_revisions_undeletable BEFORE DELETE ON result_revisions
BEGIN SELECT RAISE(ABORT, 'result revisions are immutable'); END;
CREATE TRIGGER review_events_immutable BEFORE UPDATE ON review_events
BEGIN SELECT RAISE(ABORT, 'review events are append-only'); END;
"""


def connect(path: Path) -> sqlite3.Connection:
    """Open the database in WAL mode with foreign keys on; create or migrate the schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA synchronous = NORMAL")
    _migrate(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` ... ``COMMIT``, rolled back on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    return 0 if row is None else int(row["version"])


def _migrate(conn: sqlite3.Connection) -> None:
    current = _version(conn)
    if current > SCHEMA_VERSION:
        raise RuntimeError(f"database schema {current} is newer than this code ({SCHEMA_VERSION})")
    if current < 1:
        try:
            # executescript runs trigger bodies correctly; the script is one transaction.
            conn.executescript(
                "BEGIN IMMEDIATE;\n"
                + _SCHEMA_V1
                + "\nDELETE FROM schema_version;"
                + "\nINSERT INTO schema_version (version) VALUES (1);"
                + "\nCOMMIT;"
            )
        except sqlite3.OperationalError as exc:
            # Another process created the schema first.
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if "already exists" not in str(exc) or _version(conn) < 1:
                raise
    _apply(conn, 2, _SCHEMA_V2)
    _apply(conn, 3, _SCHEMA_V3)


def _apply(conn: sqlite3.Connection, version: int, script: str) -> None:
    """Apply one schema step if the database is older; tolerate a concurrent migration."""
    if _version(conn) >= version:
        return
    try:
        conn.executescript(
            "BEGIN IMMEDIATE;\n"
            + script
            + f"\nUPDATE schema_version SET version = {version};"
            + "\nCOMMIT;"
        )
    except sqlite3.OperationalError as exc:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        concurrent = "already exists" in str(exc) or "duplicate column" in str(exc)
        if not concurrent or _version(conn) < version:
            raise
