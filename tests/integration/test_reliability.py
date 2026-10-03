"""Reliability at scale: many jobs, injected failures, two workers, no duplicates (Stage 10).

The design target is that at least 99% of 100 valid jobs complete and that retries never
duplicate a result. Here 100 distinct recordings are queued; two workers process them at the
same time; and a third of the jobs fail once at a chosen point: before any artifact is
written, after the artifacts but before publication, or just after publication (a crash
between committing the result and reporting success). Every job must end with exactly one
result revision, one artifact directory, and counts equal to an undisturbed run.
"""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from passagewatch.service.api.app import create_app
from passagewatch.service.bundle import ReleaseBundle
from passagewatch.service.db import connect
from passagewatch.service.jobs import JobStore
from passagewatch.service.settings import ServiceSettings
from passagewatch.service.worker import loop
from passagewatch.service.worker.loop import run_worker
from passagewatch.service.worker.pipeline import InferencePipeline

from .fakes import BUNDLE, BrightBoxDetector, H, W

JOBS = 100
METERS = {"x_meter_start": -1.0, "x_meter_stop": 1.0, "y_meter_start": 3.0, "y_meter_stop": 0.0}


def recording(i: int) -> bytes:
    """A distinct 12-frame recording: target crossing left to right on row band i % 40."""
    buffer = io.BytesIO()
    y = 4 + (i % 40)
    with zipfile.ZipFile(buffer, "w") as zf:
        for t in range(12):
            image = np.full((H, W), 40 + i % 7, dtype=np.uint8)
            x = 2 + 3 * t
            image[y : y + 6, x : x + 8] = 240
            zf.writestr(f"{t}.png", bytes(cv2.imencode(".png", image)[1]))
    return buffer.getvalue()


class Injected(RuntimeError):
    pass


def test_retries_and_concurrent_workers_never_duplicate_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "bundle.json").write_text(json.dumps(BUNDLE))
    settings = ServiceSettings(
        data_dir=tmp_path / "var",
        bundle_dir=bundle_dir,
        max_queue=JOBS + 10,
        worker_poll_seconds=0.01,
        heartbeat_seconds=0.2,
        lease_seconds=5,
    )

    with TestClient(create_app(settings)) as client:
        job_ids = []
        for i in range(JOBS):
            clip = client.post(
                "/v1/clips",
                files={"file": ("clip.zip", recording(i))},
                data={k: str(v) for k, v in (METERS | {"framerate": 10}).items()},
            )
            assert clip.status_code == 201, clip.text
            job = client.post("/v1/jobs", json={"clip_id": clip.json()["clip_id"], "counting": {}})
            assert job.status_code == 202, job.text
            job_ids.append(job.json()["job_id"])

    # One failure each for a third of the jobs, at three different points.
    before_artifacts = set(job_ids[0::9])
    before_publish = set(job_ids[3::9])
    after_publish = set(job_ids[6::9])
    failed_once: set[str] = set()
    lock = threading.Lock()

    def first_time(job_id: str, group: set[str]) -> bool:
        with lock:
            if job_id in group and job_id not in failed_once:
                failed_once.add(job_id)
                return True
            return False

    real_write = loop.write_track_artifacts
    real_publish = JobStore.publish_result

    def write(job_dir: Path, *args: Any, **kwargs: Any) -> None:
        if first_time(job_dir.name, before_artifacts):
            raise Injected("before artifacts")
        real_write(job_dir, *args, **kwargs)
        if first_time(job_dir.name, before_publish):
            raise Injected("after artifacts, before publication")

    def publish(self: JobStore, job_id: str, *args: Any, **kwargs: Any) -> Any:
        result = real_publish(self, job_id, *args, **kwargs)
        if first_time(job_id, after_publish):
            raise Injected("after publication")
        return result

    monkeypatch.setattr(loop, "write_track_artifacts", write)
    monkeypatch.setattr(JobStore, "publish_result", publish)

    stop = threading.Event()
    detectors = {name: BrightBoxDetector() for name in ("w1", "w2")}

    def worker(name: str) -> None:
        pipeline = InferencePipeline(
            ReleaseBundle.model_validate(BUNDLE), detectors[name], batch_size=4
        )
        run_worker(settings, pipeline, name, stop=stop)

    threads = [threading.Thread(target=worker, args=(name,)) for name in detectors]
    for t in threads:
        t.start()
    conn = connect(settings.db_path)
    deadline = time.monotonic() + 120
    try:
        while time.monotonic() < deadline:
            done = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('succeeded', 'failed')"
            ).fetchone()[0]
            if done == JOBS:
                break
            time.sleep(0.1)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=30)

    # Both workers really ran jobs (each detector saw frames).
    assert all(d.channels for d in detectors.values())
    statuses = dict(conn.execute("SELECT job_id, status FROM jobs").fetchall())
    assert sum(s == "succeeded" for s in statuses.values()) == JOBS
    assert failed_once == before_artifacts | before_publish | after_publish

    revisions = conn.execute(
        "SELECT job_id, COUNT(*), MIN(revision), MAX(revision) FROM result_revisions"
        " GROUP BY job_id"
    ).fetchall()
    assert len(revisions) == JOBS
    assert all(tuple(r[1:]) == (1, 0, 0) for r in revisions)  # one automatic revision each
    assert conn.execute("SELECT COUNT(*) FROM review_events").fetchone()[0] == 0
    artifact_dirs = sorted(p.name for p in settings.artifacts_dir.iterdir())
    assert artifact_dirs == sorted(job_ids)

    # Every job, retried or not, has the undisturbed counts: one rightward passage.
    counts = [json.loads(c) for (c,) in conn.execute("SELECT counts_json FROM result_revisions")]
    assert all((c["right"], c["left"]) == (1, 0) for c in counts)
    attempts = dict(conn.execute("SELECT job_id, attempts FROM jobs").fetchall())
    assert all(attempts[j] == 2 for j in before_artifacts | before_publish)
    assert all(attempts[j] == 1 for j in after_publish)  # published before the crash
