"""Review, export, frames and overlays: the API the review interface uses."""

from __future__ import annotations

import csv
import io
import sqlite3
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from passagewatch.service.api.app import create_app
from passagewatch.service.db import connect
from passagewatch.service.settings import ServiceSettings

from .test_api import complete, new_clip, settings_for


@pytest.fixture
def settings(tmp_path: Any) -> ServiceSettings:
    return settings_for(tmp_path)


@pytest.fixture
def client(settings: ServiceSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def finished_job(client: TestClient, settings: ServiceSettings, **counting: Any) -> str:
    """A succeeded job with tracks 1, 2 (right), 3 (left) and 4 (no passage)."""
    job_id = client.post(
        "/v1/jobs", json={"clip_id": new_clip(client), "counting": counting}
    ).json()["job_id"]
    complete(settings, job_id, right=2, left=1)
    return str(job_id)


def review(client: TestClient, job_id: str, **body: Any) -> Any:
    return client.post(f"/v1/jobs/{job_id}/reviews", json=body)


def test_reviews_create_revisions_and_keep_automatic_counts(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = finished_job(client, settings, upstream_direction="right")

    first = review(client, job_id, base_revision=0, action="reject", track_id=1, reason="debris")
    second = review(
        client, job_id, base_revision=1, action="set_direction", track_id=3, direction="right"
    )

    assert first.status_code == 201 and first.json()["revision"] == 1
    assert first.json()["reviewed"]["right"] == 1 and first.json()["reviewed"]["left"] == 1
    assert second.json()["revision"] == 2
    results = client.get(f"/v1/jobs/{job_id}/results").json()
    assert results["automatic"] == {
        "right": 2,
        "left": 1,
        "upstream": 2,
        "downstream": 1,
        "net_upstream": 1,
    }
    assert results["reviewed"] == {
        "right": 2,
        "left": 0,
        "upstream": 2,
        "downstream": 0,
        "net_upstream": 2,
    }
    assert results["review"] == {"revision": 2, "kind": "reviewed", "state": "reviewed"}
    tracks = {t["track_id"]: t for t in client.get(f"/v1/jobs/{job_id}/tracks").json()["tracks"]}
    assert (tracks[1]["review_state"], tracks[1]["final_direction"]) == ("rejected", None)
    assert (tracks[3]["review_state"], tracks[3]["final_direction"]) == ("corrected", "right")
    assert (tracks[2]["review_state"], tracks[2]["final_direction"]) == ("automatic", "right")


def test_a_stale_revision_is_rejected_with_409(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = finished_job(client, settings)
    # Two reviewers both start from revision 0; the second to save is refused.
    assert review(client, job_id, base_revision=0, action="accept", track_id=2).status_code == 201

    stale = review(client, job_id, base_revision=0, action="reject", track_id=2)

    assert stale.status_code == 409 and "latest is 1" in stale.text
    assert client.get(f"/v1/jobs/{job_id}/results").json()["review"]["revision"] == 1


def test_unresolved_tracks_are_excluded_and_counted(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = finished_job(client, settings)

    body = review(client, job_id, base_revision=0, action="mark_unresolved", track_id=2).json()

    assert (body["reviewed"]["right"], body["unresolved"]) == (1, 1)


def test_missed_fish_can_be_added_and_removed(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = finished_job(client, settings)

    added = review(
        client, job_id, base_revision=0, action="add_passage", direction="left", frame_index=2
    )
    passage = client.get(f"/v1/jobs/{job_id}/export").json()["added_passages"][0]
    removed = review(
        client, job_id, base_revision=1, action="reject", passage_id=passage["passage_id"]
    )

    assert added.json()["reviewed"]["left"] == 2 and added.json()["added_passages"] == 1
    assert passage["frame_index"] == 2 and passage["time_s"] == pytest.approx(0.2)
    assert removed.json()["reviewed"]["left"] == 1 and removed.json()["added_passages"] == 0


@pytest.mark.parametrize(
    "body",
    [
        {"action": "reject", "track_id": 99},
        {"action": "add_passage", "direction": "left"},
        {"action": "add_passage", "direction": "left", "frame_index": 5000},
        {"action": "reject", "passage_id": "p999"},
        {"action": "set_direction", "track_id": 1, "direction": "north"},
        {"action": "rename", "track_id": 1},
    ],
)
def test_invalid_reviews_are_rejected(
    client: TestClient, settings: ServiceSettings, body: dict[str, Any]
) -> None:
    job_id = finished_job(client, settings)

    response = review(client, job_id, base_revision=0, **body)

    assert response.status_code == 422
    assert client.get(f"/v1/jobs/{job_id}/reviews").json()["events"] == []


def test_unfinished_or_unknown_jobs_cannot_be_reviewed(client: TestClient) -> None:
    job_id = client.post("/v1/jobs", json={"clip_id": new_clip(client)}).json()["job_id"]

    assert review(client, job_id, base_revision=0, action="accept", track_id=1).status_code == 409
    assert (
        review(client, "job_nope", base_revision=0, action="accept", track_id=1).status_code == 404
    )


def test_review_history_is_append_only(client: TestClient, settings: ServiceSettings) -> None:
    job_id = finished_job(client, settings)
    review(client, job_id, base_revision=0, action="accept", track_id=1)
    review(client, job_id, base_revision=1, action="reject", track_id=4, reason="noise")

    events = client.get(f"/v1/jobs/{job_id}/reviews").json()["events"]

    assert [(e["action"], e["resulting_revision"]) for e in events] == [
        ("accept", 1),
        ("reject", 2),
    ]
    assert events[1]["reason"] == "noise"
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        connect(settings.db_path).execute("UPDATE review_events SET payload_json = '{}'")


def test_export_json_traces_counts_to_recording_model_and_review(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = finished_job(client, settings, upstream_direction="left", line_x_normalized=0.45)
    review(client, job_id, base_revision=0, action="mark_unresolved", track_id=1)
    review(client, job_id, base_revision=1, action="reject", track_id=3)

    latest = client.get(f"/v1/jobs/{job_id}/export")
    earlier = client.get(f"/v1/jobs/{job_id}/export", params={"revision": 1}).json()

    assert (
        latest.headers["content-disposition"]
        == f'attachment; filename="passagewatch_{job_id}_r2.json"'
    )
    report = latest.json()
    assert report["recording"]["clip_id"] and len(report["recording"]["sha256"]) == 64
    assert report["counting"] == {
        "policy": "cfc-compatible-v1",
        "line_x_normalized": 0.45,
        "upstream_direction": "left",
    }
    assert report["pipeline"]["pipeline_version"] == "pw-test-1"
    assert report["pipeline"]["detector"]["checkpoint_sha256"] == "d" * 64
    assert (
        report["counts"]["automatic"]["right"] == 2
        and report["counts"]["automatic"]["downstream"] == 2
    )
    assert report["counts"]["reviewed"]["right"] == 1 and report["counts"]["reviewed"]["left"] == 0
    assert [u["track_id"] for u in report["unresolved"]] == [1]
    track = report["tracks"][0]
    assert {"start_frame", "end_frame", "start_time_s", "end_time_s"} <= set(track)
    assert len(report["review_events"]) == 2
    assert earlier["revision"] == 1 and len(earlier["review_events"]) == 1
    assert earlier["counts"]["reviewed"]["left"] == 1  # before track 3 was rejected


def test_export_before_review_has_no_reviewed_counts(
    client: TestClient, settings: ServiceSettings
) -> None:
    report = client.get(f"/v1/jobs/{finished_job(client, settings)}/export").json()

    assert report["revision"] == 0 and report["counts"]["reviewed"] is None
    assert (
        client.get(f"/v1/jobs/{report['job']['job_id']}/export", params={"revision": 7}).status_code
        == 404
    )


def test_export_csv_has_metadata_lines_and_one_row_per_case(
    client: TestClient, settings: ServiceSettings
) -> None:
    job_id = finished_job(client, settings)
    review(client, job_id, base_revision=0, action="add_passage", direction="right", frame_index=1)

    response = client.get(f"/v1/jobs/{job_id}/export", params={"format": "csv"})

    assert response.headers["content-type"].startswith("text/csv")
    lines = response.text.splitlines()
    meta = dict(line[2:].split(",", 1) for line in lines if line.startswith("# "))
    assert meta["pipeline_version"] == "pw-test-1" and meta["upstream_direction"] == "not set"
    assert (
        meta["automatic_right"] == "2" and meta["reviewed_right"] == "3" and meta["revision"] == "1"
    )
    rows = list(
        csv.DictReader(io.StringIO("\n".join(line for line in lines if not line.startswith("# "))))
    )
    assert [r["kind"] for r in rows] == ["track"] * 4 + ["added_passage"]
    assert rows[-1]["final_direction"] == "right" and rows[0]["review_state"] == "automatic"


def test_frames_are_served_for_display(client: TestClient) -> None:
    clip_id = new_clip(client)

    frame = client.get(f"/v1/clips/{clip_id}/frames/0")
    info = client.get(f"/v1/clips/{clip_id}").json()

    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
    image = cv2.imdecode(np.frombuffer(frame.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (60, 40) and info["num_frames"] == 5
    assert client.get(f"/v1/clips/{clip_id}/frames/5").status_code == 404
    client.delete(f"/v1/clips/{clip_id}")
    assert client.get(f"/v1/clips/{clip_id}/frames/0").status_code == 410


def test_observations_cover_a_frame_window(client: TestClient, settings: ServiceSettings) -> None:
    job_id = finished_job(client, settings)

    window = client.get(f"/v1/jobs/{job_id}/observations", params={"start": 1, "stop": 3}).json()

    assert len(window["boxes"]) == 4 * 2  # four tracks, two frames
    assert [b["frame_index"] for b in window["boxes"][:4]] == [1, 1, 1, 1]
    too_wide = client.get(f"/v1/jobs/{job_id}/observations", params={"start": 0, "stop": 1000})
    assert too_wide.status_code == 422


def test_reviews_of_a_cached_job_are_its_own(client: TestClient, settings: ServiceSettings) -> None:
    clip_id = new_clip(client)
    original = client.post("/v1/jobs", json={"clip_id": clip_id}).json()["job_id"]
    complete(settings, original)
    cached = client.post("/v1/jobs", json={"clip_id": clip_id}).json()["job_id"]

    review(client, original, base_revision=0, action="reject", track_id=1)

    assert client.get(f"/v1/jobs/{cached}/results").json()["review"]["revision"] == 0
    assert review(client, cached, base_revision=0, action="accept", track_id=1).status_code == 201
    assert client.get(f"/v1/jobs/{original}/results").json()["reviewed"]["right"] == 1
