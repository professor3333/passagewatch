from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from passagewatch.counting.policy import Direction, Outcome, TrajectoryCount
from passagewatch.service.api.app import create_app
from passagewatch.service.artifacts import write_track_artifacts
from passagewatch.service.catalog import worker_heartbeat
from passagewatch.service.db import connect
from passagewatch.service.jobs import JobStore, utc_now
from passagewatch.service.settings import ServiceSettings
from passagewatch.tracking.kalman import Trajectory

BUNDLE = {
    "pipeline_version": "pw-test-1",
    "detector": {
        "model_size": "tiny",
        "checkpoint": "detector.pt",
        "checkpoint_sha256": "d" * 64,
        "input_height": 960,
        "input_width": 416,
        "score_threshold": 0.2,
        "training_run": "yolox-tiny-v1",
        "epoch": 25,
    },
    "preprocessing_version": "letterbox-gray3-v1",
    "tracker": {"kind": "kalman", "config": {"gate_m": 0.5}},
    "counting_policy": "cfc-compatible-v1",
    "provenance": {"note": "test"},
}
METERS = {"x_meter_start": -1.0, "x_meter_stop": 1.0, "y_meter_start": 5.0, "y_meter_stop": 0.5}


def settings_for(tmp_path: Path, *, bundle: bool = True, **overrides: Any) -> ServiceSettings:
    bundle_dir = tmp_path / "bundle"
    if bundle:
        bundle_dir.mkdir()
        (bundle_dir / "bundle.json").write_text(json.dumps(BUNDLE))
    return ServiceSettings(data_dir=tmp_path / "var", bundle_dir=bundle_dir, **overrides)


@pytest.fixture
def settings(tmp_path: Path) -> ServiceSettings:
    return settings_for(tmp_path, max_queue=3, max_upload_bytes=2_000_000, max_frames=50)


@pytest.fixture
def client(settings: ServiceSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def frames_zip(
    n: int = 5, *, names: list[str] | None = None, extra: dict[str, bytes] | None = None
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i, name in enumerate(names or [f"clip/{i}.jpg" for i in range(n)]):
            image = np.full((60, 40), 50 + i, dtype=np.uint8)
            zf.writestr(name, cv2.imencode(".jpg", image)[1].tobytes())
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buffer.getvalue()


def upload(client: TestClient, data: bytes, filename: str = "clip.zip", **fields: Any) -> Any:
    form = {k: str(v) for k, v in ({**METERS, "framerate": 10.0} | fields).items() if v is not None}
    return client.post("/v1/clips", files={"file": (filename, data)}, data=form)


def new_clip(client: TestClient) -> str:
    response = upload(client, frames_zip())
    assert response.status_code == 201, response.text
    return str(response.json()["clip_id"])


def complete(settings: ServiceSettings, job_id: str, right: int = 2, left: int = 1) -> None:
    """Stand-in for the worker: lease, write artifacts, publish."""
    store = JobStore(connect(settings.db_path))
    job = store.lease("test-worker", lease_seconds=60, now=utc_now())
    assert job is not None and job.job_id == job_id
    trajectories, counts = [], []
    for i in range(right + left + 1):
        direction = Direction.RIGHT if i < right else Direction.LEFT if i < right + left else None
        trajectories.append(
            Trajectory(
                i + 1, np.arange(3), np.tile([1.0, 2, 11, 12], (3, 1)), np.array([0.9, 0.5, 0.7])
            )
        )
        counts.append(
            TrajectoryCount(
                i + 1,
                0,
                2,
                3,
                0.2,
                0.8,
                0.6,
                Outcome.PASSAGE if direction else Outcome.STATIONARY,
                direction,
            )
        )
    write_track_artifacts(settings.artifacts_dir / job_id, trajectories, counts, framerate=10.0)
    store.publish_result(
        job_id,
        "test-worker",
        counts={"right": right, "left": left, "tracks": len(trajectories)},
        tracks_artifact=job_id,
        now=utc_now(),
    )


# -- health and release ------------------------------------------------------------------


def test_health_and_model_info(client: TestClient, settings: ServiceSettings) -> None:
    assert client.get("/health/live").json() == {"status": "ok"}
    not_ready = client.get("/health/ready")
    assert not_ready.status_code == 503 and not_ready.json()["checks"]["worker"] is False

    worker_heartbeat(connect(settings.db_path), "w1", "pw-test-1", now=utc_now())

    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["checks"] == {
        "database": True,
        "storage": True,
        "release": True,
        "worker": True,
    }
    info = client.get("/v1/model-info").json()
    assert info["pipeline_version"] == "pw-test-1"
    assert info["detector"]["checkpoint_sha256"] == "d" * 64
    assert (
        info["counting_policy"] == "cfc-compatible-v1" and len(info["pipeline_config_sha256"]) == 64
    )


def test_without_a_release_the_service_is_not_ready(tmp_path: Path) -> None:
    with TestClient(create_app(settings_for(tmp_path, bundle=False))) as c:
        assert c.get("/health/ready").status_code == 503
        assert c.get("/v1/model-info").status_code == 503
        clip_id = new_clip(c)
        assert c.post("/v1/jobs", json={"clip_id": clip_id}).status_code == 503


# -- uploads ---------------------------------------------------------------------------


def test_frame_zip_upload_is_registered(client: TestClient, settings: ServiceSettings) -> None:
    data = frames_zip(5, extra={"__MACOSX/clip/._0.jpg": b"x", "clip/.DS_Store": b"x"})

    response = upload(client, data)

    assert response.status_code == 201, response.text
    body = response.json()
    assert response.headers["location"] == f"/v1/clips/{body['clip_id']}"
    assert body["sha256"] == hashlib.sha256(data).hexdigest()
    assert (body["media_kind"], body["num_frames"], body["width"], body["height"]) == (
        "frames",
        5,
        40,
        60,
    )
    assert body["duration_seconds"] == 0.5 and body["expires_at"] is not None
    assert body["meters"] == {"x_start": -1.0, "x_stop": 1.0, "y_start": 5.0, "y_stop": 0.5}
    assert any((settings.media_dir / body["clip_id"]).iterdir())


def test_video_upload_takes_the_container_framerate(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 8.0, (64, 96))
    for i in range(12):
        writer.write(np.full((96, 64, 3), 40 + 5 * i, dtype=np.uint8))
    writer.release()

    response = upload(client, path.read_bytes(), "clip.mp4", framerate=None)

    assert response.status_code == 201, response.text
    body = response.json()
    assert (body["media_kind"], body["num_frames"], body["framerate"]) == ("video", 12, 8.0)
    assert (body["width"], body["height"]) == (64, 96)


@pytest.mark.parametrize(
    ("data", "fields", "message"),
    [
        (frames_zip(names=["0.jpg", "1.jpg", "3.jpg"]), {}, "without gaps"),
        (frames_zip(2, extra={"clip/notes.txt": b"x"}), {}, "not a frame file"),
        (frames_zip(names=["a/0.jpg", "b/1.jpg"]), {}, "one folder"),
        (frames_zip(3), {"framerate": None}, "framerate is required"),
        (frames_zip(3), {"x_meter_stop": -1.0}, "non-zero extent"),
        (frames_zip(51), {}, "exceed the limit of 50"),
        (b"not a recording", {}, "not a ZIP of frames or a readable video"),
        (b"", {}, "empty"),
    ],
)
def test_invalid_uploads_are_rejected(
    client: TestClient, settings: ServiceSettings, data: bytes, fields: dict[str, Any], message: str
) -> None:
    response = upload(client, data, **fields)

    assert response.status_code == 422 and message in response.text
    assert not any(settings.media_dir.iterdir())


def test_oversized_upload_is_rejected(client: TestClient, settings: ServiceSettings) -> None:
    response = upload(client, b"x" * 3_000_000)

    assert response.status_code == 413
    assert not any(settings.media_dir.iterdir())


# -- jobs --------------------------------------------------------------------------------


def test_job_creation_returns_202_at_once(client: TestClient) -> None:
    clip_id = new_clip(client)

    response = client.post("/v1/jobs", json={"clip_id": clip_id})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued" and body["pipeline_version"] == "pw-test-1"
    assert response.headers["location"] == body["status_url"] == f"/v1/jobs/{body['job_id']}"
    job = client.get(body["status_url"]).json()
    assert job["counting"] == {
        "policy": "cfc-compatible-v1",
        "line_x_normalized": 0.5,
        "upstream_direction": None,
    }
    assert job["results_url"] is None


def test_idempotency_key_replays_and_conflicts(client: TestClient) -> None:
    clip_id = new_clip(client)
    headers = {"Idempotency-Key": "upload-42"}

    first = client.post("/v1/jobs", json={"clip_id": clip_id}, headers=headers)
    again = client.post("/v1/jobs", json={"clip_id": clip_id}, headers=headers)
    conflict = client.post(
        "/v1/jobs",
        json={"clip_id": clip_id, "counting": {"line_x_normalized": 0.3}},
        headers=headers,
    )

    assert first.status_code == again.status_code == 202
    assert again.json()["job_id"] == first.json()["job_id"]
    assert conflict.status_code == 409


def test_full_queue_answers_503_with_retry_after(client: TestClient) -> None:
    clip_id = new_clip(client)
    for line in (0.3, 0.4, 0.5):
        client.post("/v1/jobs", json={"clip_id": clip_id, "counting": {"line_x_normalized": line}})

    response = client.post(
        "/v1/jobs", json={"clip_id": clip_id, "counting": {"line_x_normalized": 0.6}}
    )

    assert response.status_code == 503 and response.headers["retry-after"] == "30"


@pytest.mark.parametrize(
    "body",
    [
        {"counting": {"line_x_normalized": 1.2}},
        {"counting": {"policy": "my-own-policy"}},
        {"counting": {"upstream_direction": "north"}},
        {"pipeline_alias": "experimental"},
        {"unexpected": True},
    ],
)
def test_invalid_job_requests_are_rejected(client: TestClient, body: dict[str, Any]) -> None:
    clip_id = new_clip(client)

    assert client.post("/v1/jobs", json={"clip_id": clip_id, **body}).status_code == 422


def test_unknown_and_deleted_clips(client: TestClient) -> None:
    assert client.post("/v1/jobs", json={"clip_id": "clip_nope"}).status_code == 404
    clip_id = new_clip(client)
    assert client.delete(f"/v1/clips/{clip_id}").status_code == 204

    assert client.post("/v1/jobs", json={"clip_id": clip_id}).status_code == 410
    assert client.get("/v1/jobs/job_nope").status_code == 404


# -- results and tracks ---------------------------------------------------------------------


def test_results_are_unavailable_until_the_job_succeeds(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", json={"clip_id": new_clip(client)}).json()["job_id"]

    assert client.get(f"/v1/jobs/{job_id}/results").status_code == 409
    assert client.get(f"/v1/jobs/{job_id}/tracks").status_code == 409


def test_results_keep_automatic_counts_and_map_river_directions(
    client: TestClient, settings: ServiceSettings
) -> None:
    clip_id = new_clip(client)
    job_id = client.post(
        "/v1/jobs", json={"clip_id": clip_id, "counting": {"upstream_direction": "left"}}
    ).json()["job_id"]
    complete(settings, job_id, right=2, left=1)

    body = client.get(f"/v1/jobs/{job_id}/results").json()

    assert body["automatic"] == {
        "right": 2,
        "left": 1,
        "upstream": 1,
        "downstream": 2,
        "net_upstream": -1,
    }
    assert body["reviewed"] is None
    assert body["review"] == {"revision": 0, "kind": "automatic", "state": "pending"}
    assert body["cached"] is False and body["tracks"] == 4
    assert body["pipeline_version"] == "pw-test-1" and len(body["recording_sha256"]) == 64
    assert client.get(f"/v1/jobs/{job_id}").json()["results_url"] == f"/v1/jobs/{job_id}/results"


def test_without_orientation_only_image_directions_are_reported(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = client.post("/v1/jobs", json={"clip_id": new_clip(client)}).json()["job_id"]
    complete(settings, job_id)

    automatic = client.get(f"/v1/jobs/{job_id}/results").json()["automatic"]

    assert automatic == {
        "right": 2,
        "left": 1,
        "upstream": None,
        "downstream": None,
        "net_upstream": None,
    }


def test_tracks_are_paginated(client: TestClient, settings: ServiceSettings) -> None:
    job_id = client.post("/v1/jobs", json={"clip_id": new_clip(client)}).json()["job_id"]
    complete(settings, job_id, right=2, left=1)

    page = client.get(f"/v1/jobs/{job_id}/tracks", params={"offset": 1, "limit": 2}).json()

    assert page["total"] == 4 and [t["track_id"] for t in page["tracks"]] == [2, 3]
    assert page["tracks"][1]["direction"] == "left" and page["tracks"][1]["start_time_s"] == 0.0
    too_many = client.get(
        f"/v1/jobs/{job_id}/tracks", params={"limit": settings.tracks_page_limit + 1}
    )
    assert too_many.status_code == 422


def test_identical_analysis_is_served_from_cache(
    client: TestClient, settings: ServiceSettings
) -> None:
    clip_id = new_clip(client)
    first = client.post("/v1/jobs", json={"clip_id": clip_id}).json()["job_id"]
    complete(settings, first)

    second = client.post("/v1/jobs", json={"clip_id": clip_id}).json()

    assert second["status"] == "succeeded" and second["job_id"] != first
    results = client.get(f"/v1/jobs/{second['job_id']}/results").json()
    assert results["cached"] is True and results["automatic"]["right"] == 2


def test_clip_with_active_jobs_cannot_be_deleted(
    client: TestClient, settings: ServiceSettings
) -> None:
    clip_id = new_clip(client)
    job_id = client.post("/v1/jobs", json={"clip_id": clip_id}).json()["job_id"]

    assert client.delete(f"/v1/clips/{clip_id}").status_code == 409
    complete(settings, job_id)
    assert client.delete(f"/v1/clips/{clip_id}").status_code == 204
    assert client.delete(f"/v1/clips/{clip_id}").status_code == 204  # idempotent
    assert client.delete("/v1/clips/clip_nope").status_code == 404
    assert not (settings.media_dir / clip_id).exists()
    assert client.get(f"/v1/jobs/{job_id}/results").status_code == 200  # provenance kept
