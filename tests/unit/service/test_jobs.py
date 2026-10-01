from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from passagewatch.service.catalog import (
    PipelineVersionConflictError,
    add_clip,
    expired_clips,
    get_clip,
    mark_clip_deleted,
    register_pipeline_version,
)
from passagewatch.service.db import connect
from passagewatch.service.jobs import (
    FAILED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    IdempotencyConflictError,
    JobStore,
    LeaseLostError,
    QueueFullError,
)

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
COUNTING = {"policy": "cfc-compatible-v1", "line_x_normalized": 0.5, "upstream_direction": None}
LEASE = 60


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    return connect(tmp_path / "service.db")


@pytest.fixture
def store(conn: sqlite3.Connection) -> JobStore:
    register_pipeline_version(conn, "pw-test", {"detector": "x"}, now=T0)
    for clip_id, sha in (("clip_a", "a" * 64), ("clip_b", "b" * 64)):
        add_clip(
            conn,
            clip_id=clip_id,
            source="upload",
            sha256=sha,
            media_kind="frames",
            media_path=f"media/{clip_id}",
            num_frames=100,
            width=288,
            height=624,
            framerate=10.0,
            meters=(-1.0, 1.0, 5.0, 0.5),
            now=T0,
            expires_at=at(86400),
        )
    return JobStore(conn, max_queue=3, max_attempts=2, deadline_seconds=600)


def create(store: JobStore, clip: str = "clip_a", key: str | None = None, **counting: object):  # type: ignore[no-untyped-def]
    return store.create(
        clip_id=clip,
        clip_sha256=clip[-1] * 64,
        pipeline_version="pw-test",
        pipeline_config_sha256="c" * 64,
        counting=COUNTING | counting,
        idempotency_key=key,
        now=T0,
    )


def run_to_success(store: JobStore, job_id: str, worker: str = "w1") -> None:
    leased = store.lease(worker, lease_seconds=LEASE, now=at(1))
    assert leased is not None and leased.job_id == job_id
    store.publish_result(
        job_id, worker, counts={"right": 1}, tracks_artifact="a.parquet", now=at(2)
    )


# -- creation ---------------------------------------------------------------------------


def test_new_jobs_are_queued(store: JobStore) -> None:
    created = create(store)

    assert created.created and created.job.status == QUEUED
    assert created.job.counting == COUNTING and created.job.attempts == 0


def test_same_idempotency_key_and_request_returns_the_same_job(store: JobStore) -> None:
    first = create(store, key="k1")
    again = create(store, key="k1")

    assert not again.created and again.job.job_id == first.job.job_id


def test_idempotency_key_reused_for_another_request_is_refused(store: JobStore) -> None:
    create(store, key="k1")

    with pytest.raises(IdempotencyConflictError):
        create(store, key="k1", line_x_normalized=0.4)


def test_queue_is_bounded(store: JobStore) -> None:
    for line in (0.3, 0.4, 0.5):
        create(store, line_x_normalized=line)

    with pytest.raises(QueueFullError):
        create(store, line_x_normalized=0.6)


def test_identical_analysis_is_answered_from_the_result_cache(store: JobStore) -> None:
    original = create(store).job
    run_to_success(store, original.job_id)

    cached = create(store).job
    different = create(store, line_x_normalized=0.4).job

    assert cached.status == SUCCEEDED and cached.cached_from == original.job_id
    assert store.result(cached.job_id).counts == {"right": 1}  # type: ignore[union-attr]
    assert different.status == QUEUED and different.cached_from is None


# -- leasing, heartbeats, recovery ------------------------------------------------------


def test_one_job_is_leased_to_one_worker(store: JobStore) -> None:
    job = create(store).job

    first = store.lease("w1", lease_seconds=LEASE, now=at(1))
    second = store.lease("w2", lease_seconds=LEASE, now=at(2))

    assert first is not None and first.job_id == job.job_id and first.status == RUNNING
    assert first.attempts == 1 and first.lease_owner == "w1"
    assert second is None


def test_jobs_are_leased_oldest_first(store: JobStore) -> None:
    a = store.create(
        clip_id="clip_a",
        clip_sha256="a" * 64,
        pipeline_version="pw-test",
        pipeline_config_sha256="c" * 64,
        counting=COUNTING,
        idempotency_key=None,
        now=at(0),
    ).job
    b = store.create(
        clip_id="clip_b",
        clip_sha256="b" * 64,
        pipeline_version="pw-test",
        pipeline_config_sha256="c" * 64,
        counting=COUNTING,
        idempotency_key=None,
        now=at(5),
    ).job

    assert store.lease("w1", lease_seconds=LEASE, now=at(10)).job_id == a.job_id  # type: ignore[union-attr]
    assert store.lease("w2", lease_seconds=LEASE, now=at(10)).job_id == b.job_id  # type: ignore[union-attr]


def test_heartbeat_extends_the_lease_and_records_progress(store: JobStore) -> None:
    job = create(store).job
    store.lease("w1", lease_seconds=LEASE, now=at(1))

    store.heartbeat(job.job_id, "w1", lease_seconds=LEASE, progress=0.5, now=at(50))

    assert store.lease("w2", lease_seconds=LEASE, now=at(100)) is None  # still held until 110
    assert store.get(job.job_id).progress == 0.5  # type: ignore[union-attr]


def test_abandoned_job_is_recovered_by_another_worker(store: JobStore) -> None:
    job = create(store).job
    store.lease("w1", lease_seconds=LEASE, now=at(1))  # w1 dies without heartbeats

    recovered = store.lease("w2", lease_seconds=LEASE, now=at(LEASE + 2))

    assert recovered is not None and recovered.job_id == job.job_id
    assert recovered.lease_owner == "w2" and recovered.attempts == 2
    with pytest.raises(LeaseLostError):
        store.heartbeat(job.job_id, "w1", lease_seconds=LEASE, progress=0.9, now=at(LEASE + 3))


def test_job_fails_after_its_last_attempt_is_abandoned(store: JobStore) -> None:
    job = create(store).job
    store.lease("w1", lease_seconds=LEASE, now=at(1))
    store.lease("w2", lease_seconds=LEASE, now=at(LEASE + 2))  # attempt 2 of 2

    assert store.lease("w3", lease_seconds=LEASE, now=at(2 * LEASE + 4)) is None
    failed = store.get(job.job_id)
    assert failed is not None and failed.status == FAILED and "last attempt" in str(failed.error)


def test_job_past_its_deadline_fails(store: JobStore) -> None:
    job = create(store).job
    store.lease("w1", lease_seconds=LEASE, now=at(1))
    for t in range(50, 650, 50):  # keeps heartbeating, but runs too long
        store.heartbeat(job.job_id, "w1", lease_seconds=LEASE, progress=0.1, now=at(t))

    store.lease("w2", lease_seconds=LEASE, now=at(605))

    assert store.get(job.job_id).error == "deadline exceeded"  # type: ignore[union-attr]
    with pytest.raises(LeaseLostError):
        store.publish_result(job.job_id, "w1", counts={}, tracks_artifact="x", now=at(606))


# -- results and failures ---------------------------------------------------------------


def test_publication_is_once_and_only_by_the_lease_holder(store: JobStore) -> None:
    job = create(store).job
    store.lease("w1", lease_seconds=LEASE, now=at(1))

    with pytest.raises(LeaseLostError):
        store.publish_result(job.job_id, "w2", counts={}, tracks_artifact="x", now=at(2))
    store.publish_result(job.job_id, "w1", counts={"right": 2}, tracks_artifact="t", now=at(3))
    with pytest.raises(LeaseLostError):  # a retry of the same publication
        store.publish_result(job.job_id, "w1", counts={"right": 9}, tracks_artifact="t", now=at(4))

    done = store.get(job.job_id)
    assert done is not None and done.status == SUCCEEDED and done.progress == 1.0
    result = store.result(job.job_id)
    assert result is not None and result.revision == 0 and result.counts == {"right": 2}
    rows = store.conn.execute("SELECT COUNT(*) FROM result_revisions").fetchone()[0]
    assert rows == 1


def test_recovered_job_publishes_exactly_one_result(store: JobStore) -> None:
    job = create(store).job
    store.lease("w1", lease_seconds=LEASE, now=at(1))
    store.lease("w2", lease_seconds=LEASE, now=at(LEASE + 2))  # w1 presumed dead

    store.publish_result(job.job_id, "w2", counts={"right": 1}, tracks_artifact="t", now=at(70))
    with pytest.raises(LeaseLostError):  # w1 wakes up and tries to publish late
        store.publish_result(job.job_id, "w1", counts={"right": 5}, tracks_artifact="t", now=at(71))

    assert store.conn.execute("SELECT COUNT(*) FROM result_revisions").fetchone()[0] == 1


def test_retryable_failure_requeues_until_attempts_run_out(store: JobStore) -> None:
    job = create(store).job
    store.lease("w1", lease_seconds=LEASE, now=at(1))

    first = store.fail(job.job_id, "w1", error="decode error", retryable=True, now=at(2))
    store.lease("w1", lease_seconds=LEASE, now=at(3))
    second = store.fail(job.job_id, "w1", error="decode error", retryable=True, now=at(4))

    assert first.status == QUEUED and first.error == "decode error"
    assert second.status == FAILED and second.attempts == 2


def test_permanent_failure_is_not_retried(store: JobStore) -> None:
    job = create(store).job
    store.lease("w1", lease_seconds=LEASE, now=at(1))

    failed = store.fail(job.job_id, "w1", error="bad input", retryable=False, now=at(2))

    assert failed.status == FAILED and failed.attempts == 1


def test_original_results_are_immutable(store: JobStore) -> None:
    job = create(store).job
    run_to_success(store, job.job_id)

    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        store.conn.execute("UPDATE result_revisions SET counts_json = '{}'")
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        store.conn.execute("DELETE FROM result_revisions")


# -- catalog -----------------------------------------------------------------------------


def test_pipeline_versions_are_immutable(conn: sqlite3.Connection) -> None:
    register_pipeline_version(conn, "pw-1", {"a": 1}, now=T0)
    register_pipeline_version(conn, "pw-1", {"a": 1}, now=T0)  # identical: fine

    with pytest.raises(PipelineVersionConflictError):
        register_pipeline_version(conn, "pw-1", {"a": 2}, now=T0)


def test_clips_expire_and_deletion_keeps_the_record(store: JobStore) -> None:
    conn = store.conn

    assert [c.clip_id for c in expired_clips(conn, now=at(86400))] == ["clip_a", "clip_b"]
    assert mark_clip_deleted(conn, "clip_a", now=at(86401))
    assert not mark_clip_deleted(conn, "clip_a", now=at(86402))
    clip = get_clip(conn, "clip_a")
    assert clip is not None and clip.deleted_at is not None and clip.duration_seconds == 10.0
    assert [c.clip_id for c in expired_clips(conn, now=at(86500))] == ["clip_b"]


def test_schema_survives_reconnection(tmp_path: Path) -> None:
    first = connect(tmp_path / "s.db")
    register_pipeline_version(first, "pw-1", {"a": 1}, now=T0)
    first.close()

    second = connect(tmp_path / "s.db")

    assert second.execute("SELECT version FROM schema_version").fetchone()[0] == 1
    assert second.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert second.execute("SELECT COUNT(*) FROM pipeline_versions").fetchone()[0] == 1


def test_concurrent_workers_never_lease_the_same_job(tmp_path: Path) -> None:
    import threading

    db = tmp_path / "race.db"
    setup = connect(db)
    register_pipeline_version(setup, "pw-test", {"detector": "x"}, now=T0)
    add_clip(
        setup,
        clip_id="clip_a",
        source="upload",
        sha256="a" * 64,
        media_kind="frames",
        media_path="m",
        num_frames=10,
        width=10,
        height=10,
        framerate=10.0,
        meters=(0.0, 1.0, 1.0, 0.0),
        now=T0,
        expires_at=None,
    )
    seeded = JobStore(setup, max_queue=100)
    for i in range(10):
        create(seeded, line_x_normalized=0.05 + 0.08 * i)

    leased: list[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        store = JobStore(connect(db))
        while (job := store.lease(name, lease_seconds=LEASE, now=at(1))) is not None:
            with lock:
                leased.append(job.job_id)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(leased) == 10 and len(set(leased)) == 10
