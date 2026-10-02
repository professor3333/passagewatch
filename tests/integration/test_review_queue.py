"""Review scores, the review queue and random audits through the API (Stage 9)."""

from __future__ import annotations

import io
import json
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from passagewatch.service import artifacts
from passagewatch.service.api.app import create_app
from passagewatch.service.bundle import ReleaseBundle
from passagewatch.service.settings import ServiceSettings
from passagewatch.service.worker.loop import run_worker
from passagewatch.service.worker.pipeline import BundleMismatchError, InferencePipeline

from .fakes import BUNDLE, BrightBoxDetector, H, W

CALIBRATED = BUNDLE | {"pipeline_version": "pw-test-cal", "calibration_version": "review-v0"}
METERS = {"x_meter_start": -1.0, "x_meter_stop": 1.0, "y_meter_start": 3.0, "y_meter_stop": 0.0}


def clip_zip() -> bytes:
    """A target crossing left to right in frames 0-19, then 100 empty frames (10 s)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i in range(120):
            image = np.full((H, W), 40, dtype=np.uint8)
            if i < 20:
                x = 2 + int(1.5 * i)
                image[30:36, x : x + 8] = 240
            zf.writestr(f"{i}.png", bytes(cv2.imencode(".png", image)[1]))
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> ServiceSettings:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "bundle.json").write_text(json.dumps(CALIBRATED))
    return ServiceSettings(
        data_dir=tmp_path / "var",
        bundle_dir=bundle_dir,
        worker_poll_seconds=0.05,
        heartbeat_seconds=0.2,
        lease_seconds=5,
    )


@pytest.fixture
def client(settings: ServiceSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def analyzed_job(client: TestClient, settings: ServiceSettings) -> str:
    clip = client.post(
        "/v1/clips",
        files={"file": ("clip.zip", clip_zip())},
        data={k: str(v) for k, v in (METERS | {"framerate": 10}).items()},
    )
    assert clip.status_code == 201, clip.text
    job = client.post("/v1/jobs", json={"clip_id": clip.json()["clip_id"], "counting": {}})
    assert job.status_code == 202, job.text
    pipeline = InferencePipeline(
        ReleaseBundle.model_validate(CALIBRATED), BrightBoxDetector(), batch_size=4
    )
    assert run_worker(settings, pipeline, "w", stop=threading.Event(), max_jobs=1) == 1
    return str(job.json()["job_id"])


def review(client: TestClient, job_id: str, base: int, **body: Any) -> Any:
    return client.post(f"/v1/jobs/{job_id}/reviews", json={"base_revision": base, **body})


def test_tracks_carry_review_scores_and_can_be_listed_in_queue_order(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = analyzed_job(client, settings)

    tracks = client.get(f"/v1/jobs/{job_id}/tracks").json()["tracks"]
    queue = client.get(f"/v1/jobs/{job_id}/tracks", params={"order": "queue"}).json()["tracks"]

    assert len(tracks) == 1
    track = tracks[0]
    assert track["triage"] in ("suggested", "needs_review", "unresolved")
    assert 0.0 <= track["review_score"] <= 1.0 and isinstance(track["review_reasons"], list)
    assert [t["track_id"] for t in queue] == [t["track_id"] for t in tracks]
    assert client.get(f"/v1/jobs/{job_id}/tracks", params={"order": "x"}).status_code == 422
    assert client.get("/v1/model-info").json()["calibration_version"] == "review-v0"


def test_audit_windows_are_reviewed_and_reported(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = analyzed_job(client, settings)

    audit = client.get(f"/v1/jobs/{job_id}/audit").json()
    windows = audit["windows"]
    assert audit["calibration_version"] == "review-v0" and windows
    # Windows avoid the passage (frames 0-19) and its 1 s margin.
    assert all(w["start_frame"] >= 30 and w["state"] == "pending" for w in windows)

    assert review(client, job_id, 0, action="mark_audited", audit_window=0).status_code == 201
    added = review(
        client,
        job_id,
        1,
        action="add_passage",
        direction="left",
        frame_index=windows[0]["start_frame"],
        audit_window=0,
    )
    assert added.status_code == 201 and added.json()["reviewed"]["left"] == 1

    first = client.get(f"/v1/jobs/{job_id}/audit").json()["windows"][0]
    assert (first["state"], first["passages_added"]) == ("checked", 1)
    report = client.get(f"/v1/jobs/{job_id}/export").json()
    assert report["audit"]["windows_checked"] == 1
    assert report["audit"]["passages_added_in_audits"] == 1
    assert report["tracks"][0]["triage"] is not None
    csv = client.get(f"/v1/jobs/{job_id}/export", params={"format": "csv"}).text
    assert "# calibration_version,review-v0" in csv and "# audit_windows_checked,1" in csv
    header = next(line for line in csv.splitlines() if line.startswith("kind,"))
    assert header.endswith(",triage,review_score")


def test_invalid_audit_reviews_are_rejected(client: TestClient, settings: ServiceSettings) -> None:
    job_id = analyzed_job(client, settings)
    count = len(client.get(f"/v1/jobs/{job_id}/audit").json()["windows"])

    assert review(client, job_id, 0, action="mark_audited").status_code == 422
    assert review(client, job_id, 0, action="mark_audited", audit_window=count).status_code == 422
    assert review(client, job_id, 0, action="accept", track_id=1, audit_window=0).status_code == 422


def test_unknown_calibration_versions_are_refused() -> None:
    with pytest.raises(BundleMismatchError, match="unknown calibration"):
        InferencePipeline(
            ReleaseBundle.model_validate(BUNDLE | {"calibration_version": "review-v9"}),
            BrightBoxDetector(),
        )


def test_artifacts_without_review_columns_read_back_with_nulls(tmp_path: Path) -> None:
    old_fields = [f for f in artifacts.TRACKS_SCHEMA if f.name not in artifacts.REVIEW_COLUMNS]
    row = {
        "track_id": 1,
        "start_frame": 0,
        "end_frame": 9,
        "start_time_s": 0.0,
        "end_time_s": 0.9,
        "observations": 10,
        "start_u": 0.2,
        "end_u": 0.8,
        "displacement": 0.6,
        "outcome": "passage",
        "direction": "right",
        "mean_score": 0.9,
        "min_score": 0.8,
    }
    pq.write_table(
        pa.Table.from_pylist([row], schema=pa.schema(old_fields)), tmp_path / artifacts.TRACKS_FILE
    )

    (track,) = artifacts.read_all_tracks(tmp_path)

    assert track["direction"] == "right"
    assert (track["review_score"], track["triage"], track["review_reasons"]) == (None, None, None)
    assert artifacts.read_audit(tmp_path) is None
