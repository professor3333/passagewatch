"""The full path upload -> job -> worker -> results, failures, recovery, and retention."""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from passagewatch.detection.neural import build_yolox
from passagewatch.preprocessing.temporal import GRAY3, TEMPORAL3
from passagewatch.service.api.app import create_app
from passagewatch.service.bundle import BundleExistsError, activate, build_bundle, load_bundle
from passagewatch.service.db import connect
from passagewatch.service.jobs import JobStore
from passagewatch.service.settings import ServiceSettings
from passagewatch.service.worker.loop import process_job, run_worker, sweep_expired_uploads
from passagewatch.service.worker.pipeline import BundleMismatchError, InferencePipeline

from .fakes import BUNDLE, passage_zip, pipeline, run_slow_worker

METERS = {"x_meter_start": -1.0, "x_meter_stop": 1.0, "y_meter_start": 3.0, "y_meter_stop": 0.0}


def make_settings(tmp_path: Path, **overrides: Any) -> ServiceSettings:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(exist_ok=True)
    (bundle_dir / "bundle.json").write_text(json.dumps(BUNDLE))
    values: dict[str, Any] = {
        "data_dir": tmp_path / "var",
        "bundle_dir": bundle_dir,
        "worker_poll_seconds": 0.05,
        "heartbeat_seconds": 0.2,
        "lease_seconds": 5,
    }
    return ServiceSettings(**(values | overrides))


@pytest.fixture
def settings(tmp_path: Path) -> ServiceSettings:
    return make_settings(tmp_path)


@pytest.fixture
def client(settings: ServiceSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def submit(client: TestClient, data: bytes, **counting: Any) -> str:
    clip = client.post(
        "/v1/clips",
        files={"file": ("clip.zip", data)},
        data={k: str(v) for k, v in (METERS | {"framerate": 10}).items()},
    )
    assert clip.status_code == 201, clip.text
    job = client.post("/v1/jobs", json={"clip_id": clip.json()["clip_id"], "counting": counting})
    assert job.status_code == 202, job.text
    return str(job.json()["job_id"])


def work(settings: ServiceSettings, jobs: int = 1, **kwargs: Any) -> int:
    return run_worker(settings, pipeline(**kwargs), "w-test", stop=threading.Event(), max_jobs=jobs)


def test_upload_job_worker_results_end_to_end(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = submit(client, passage_zip(), upstream_direction="right")

    assert work(settings) == 1

    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["status"] == "succeeded" and job["progress"] == 1.0 and job["attempts"] == 1
    results = client.get(f"/v1/jobs/{job_id}/results").json()
    assert results["automatic"] == {
        "right": 1,
        "left": 0,
        "upstream": 1,
        "downstream": 0,
        "net_upstream": 1,
    }
    tracks = client.get(f"/v1/jobs/{job_id}/tracks").json()
    assert tracks["total"] == 1 and tracks["tracks"][0]["direction"] == "right"
    assert tracks["tracks"][0]["observations"] == 20
    assert (settings.artifacts_dir / job_id / "observations.parquet").is_file()
    assert client.get("/health/ready").status_code == 200  # the worker reported itself ready


def test_a_temporal_pipeline_encodes_every_frame_with_its_clip(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    temporal = pipeline(preprocessing=TEMPORAL3)
    with TestClient(create_app(settings)) as c:
        job_id = submit(c, passage_zip())

        assert run_worker(settings, temporal, "w-t", stop=threading.Event(), max_jobs=1) == 1

        assert c.get(f"/v1/jobs/{job_id}").json()["status"] == "succeeded"
        assert c.get(f"/v1/jobs/{job_id}/results").json()["automatic"]["right"] == 1
        tracks = c.get(f"/v1/jobs/{job_id}/tracks").json()["tracks"]
        assert tracks[0]["observations"] == 20  # every frame, including the last
    assert temporal.detector.channels == {3}  # type: ignore[attr-defined]


def test_a_bundle_must_declare_its_checkpoints_preprocessing(tmp_path: Path) -> None:
    path = build_bundle(
        checkpoint=checkpoint(tmp_path / "c.pt", preprocessing=TEMPORAL3),
        tracker_config={},
        score_threshold=0.2,
        version="pw-1",
        bundles_dir=tmp_path / "bundles",
    )
    assert InferencePipeline.load(path).bundle.preprocessing_version == TEMPORAL3
    declared = json.loads((path / "bundle.json").read_text())
    (path / "bundle.json").write_text(json.dumps(declared | {"preprocessing_version": GRAY3}))

    with pytest.raises(BundleMismatchError, match="trained with letterbox-temporal3-v1"):
        InferencePipeline.load(path)


def test_a_moved_counting_line_is_applied(client: TestClient, settings: ServiceSettings) -> None:
    # The target ends at x ~ 0.92 of the width: a line at 0.95 is never reached.
    job_id = submit(client, passage_zip(), line_x_normalized=0.95)
    work(settings)

    assert client.get(f"/v1/jobs/{job_id}/results").json()["automatic"]["right"] == 0


def test_a_corrupt_frame_fails_the_job_permanently(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = submit(client, passage_zip(corrupt=5))  # upload checks only the first frame

    work(settings)

    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["status"] == "failed" and job["attempts"] == 1 and "frame 5" in job["error"]


def test_unexpected_errors_are_retried(client: TestClient, settings: ServiceSettings) -> None:
    job_id = submit(client, passage_zip())
    broken = pipeline()

    def explode(*_: Any, **__: Any) -> Any:
        raise RuntimeError("out of memory")

    broken.run = explode  # type: ignore[method-assign]
    conn = connect(settings.db_path)
    store = JobStore(conn, max_attempts=settings.max_attempts)
    job = store.lease("w-test", lease_seconds=5, now=_now())
    assert job is not None

    assert process_job(conn, settings, broken, job, "w-test") == "queued"
    assert "out of memory" in str(store.get(job_id).error)  # type: ignore[union-attr]
    assert work(settings) == 1
    assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "succeeded"


def test_a_worker_killed_mid_job_is_recovered_without_duplicates(
    client: TestClient, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path, lease_seconds=2, heartbeat_seconds=0.3)
    job_id = submit(client, passage_zip(40))
    context = multiprocessing.get_context("spawn")
    doomed = context.Process(target=run_slow_worker, args=(settings.model_dump_json(), "w-doomed"))
    doomed.start()
    store = JobStore(connect(settings.db_path))
    deadline = time.monotonic() + 60
    while (job := store.get(job_id)) is None or job.progress == 0:  # wait until mid-job
        assert time.monotonic() < deadline, "the worker never started the job"
        time.sleep(0.1)

    doomed.kill()  # SIGKILL: no cleanup, no heartbeats
    doomed.join()
    time.sleep(settings.lease_seconds + 0.5)  # let the lease expire

    assert work(settings) == 1

    job = store.get(job_id)
    assert job is not None and job.status == "succeeded" and job.attempts == 2
    rows = store.conn.execute("SELECT COUNT(*) FROM result_revisions WHERE job_id = ?", (job_id,))
    assert rows.fetchone()[0] == 1
    assert client.get(f"/v1/jobs/{job_id}/results").json()["automatic"]["right"] == 1


def test_a_worker_that_loses_its_lease_stops_without_publishing(
    client: TestClient, tmp_path: Path
) -> None:
    # Heartbeats slower than the lease: the lease lapses and another worker takes the job.
    settings = make_settings(tmp_path, lease_seconds=1, heartbeat_seconds=1.5)
    job_id = submit(client, passage_zip(40))
    conn = connect(settings.db_path)
    store = JobStore(conn)
    job = store.lease("w-slow", lease_seconds=1, now=_now())
    assert job is not None
    outcome: dict[str, str] = {}

    def slow() -> None:
        outcome["w-slow"] = process_job(
            connect(settings.db_path), settings, pipeline(delay=0.4), job, "w-slow"
        )

    thread = threading.Thread(target=slow)
    thread.start()
    time.sleep(1.2)
    taken = JobStore(connect(settings.db_path)).lease("w-other", lease_seconds=60, now=_now())
    thread.join()

    assert taken is not None and taken.job_id == job_id
    assert outcome["w-slow"] == "lease_lost"
    assert conn.execute("SELECT COUNT(*) FROM result_revisions").fetchone()[0] == 0
    assert store.get(job_id).lease_owner == "w-other"  # type: ignore[union-attr]


def test_expired_uploads_are_deleted_unless_a_job_needs_them(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, upload_retention_hours=1e-7)
    with TestClient(create_app(settings)) as c:
        waiting_job = submit(c, passage_zip())
        idle = c.post(
            "/v1/clips",
            files={"file": ("idle.zip", passage_zip())},
            data={k: str(v) for k, v in (METERS | {"framerate": 10}).items()},
        ).json()["clip_id"]
        time.sleep(0.01)

        removed = sweep_expired_uploads(connect(settings.db_path), settings)

        assert removed == [idle]
        assert not (settings.media_dir / idle).exists()
        assert c.post("/v1/jobs", json={"clip_id": idle}).status_code == 410
        assert c.get(f"/v1/jobs/{waiting_job}").json()["status"] == "queued"


# -- release bundles ------------------------------------------------------------------------


def checkpoint(path: Path, preprocessing: str = "letterbox-gray3-v1") -> Path:
    torch.save(
        {
            "config": {"name": "run", "model_size": "tiny", "input_height": 96, "input_width": 64},
            "preprocessing_version": preprocessing,
            "model": build_yolox("tiny").state_dict(),
            "state": {"epoch": 3},
            "metadata": {"manifest": "full-v2"},
        },
        path,
    )
    return path


def test_bundles_are_built_once_and_loaded_with_their_checkpoint(tmp_path: Path) -> None:
    ckpt = checkpoint(tmp_path / "epoch-003.pt")
    tracker = {"gate_m": 0.5}

    path = build_bundle(
        checkpoint=ckpt,
        tracker_config=tracker,
        score_threshold=0.2,
        version="pw-1",
        bundles_dir=tmp_path / "bundles",
    )
    again = build_bundle(
        checkpoint=ckpt,
        tracker_config=tracker,
        score_threshold=0.2,
        version="pw-1",
        bundles_dir=tmp_path / "bundles",
    )
    bundle = load_bundle(path)

    assert again == path and bundle.detector.epoch == 3 and bundle.detector.input_width == 64
    assert bundle.provenance["training_metadata"] == {"manifest": "full-v2"}
    with pytest.raises(BundleExistsError, match="different configuration"):
        build_bundle(
            checkpoint=ckpt,
            tracker_config=tracker,
            score_threshold=0.3,
            version="pw-1",
            bundles_dir=tmp_path / "bundles",
        )
    with pytest.raises(BundleExistsError, match="different provenance"):
        build_bundle(
            checkpoint=ckpt,
            tracker_config=tracker,
            score_threshold=0.2,
            version="pw-1",
            bundles_dir=tmp_path / "bundles",
            provenance={"selection": "elsewhere"},
        )
    assert activate(tmp_path / "bundles", "pw-1").resolve() == path.resolve()
    assert InferencePipeline.load(tmp_path / "bundles" / "active").version == "pw-1"


def test_a_tampered_checkpoint_is_refused(tmp_path: Path) -> None:
    path = build_bundle(
        checkpoint=checkpoint(tmp_path / "c.pt"),
        tracker_config={},
        score_threshold=0.2,
        version="pw-1",
        bundles_dir=tmp_path / "bundles",
    )
    with (path / "detector.pt").open("ab") as fh:
        fh.write(b"tampered")

    with pytest.raises(BundleMismatchError, match="SHA-256"):
        InferencePipeline.load(path)


def test_a_bundle_for_other_preprocessing_is_refused(tmp_path: Path) -> None:
    path = build_bundle(
        checkpoint=checkpoint(tmp_path / "c.pt", preprocessing="letterbox-gray3-v0"),
        tracker_config={},
        score_threshold=0.2,
        version="pw-1",
        bundles_dir=tmp_path / "bundles",
    )

    with pytest.raises(BundleMismatchError, match="preprocessing"):
        InferencePipeline.load(path)


def _now() -> Any:
    from passagewatch.service.jobs import utc_now

    return utc_now()
