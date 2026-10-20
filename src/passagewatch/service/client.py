"""Small HTTP helpers for scripts that drive a running PassageWatch service: upload a CFC
clip's frames, create a job, wait for it, and read a reviewer's decisions back (the
usability study, the review sample, the same-fish re-test).

Development tooling only: it needs ``httpx``, a development dependency, so the service itself
never imports it."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx

from passagewatch.evaluation.review_sample import (
    TrackLabel,
    describe_changes,
    track_labels,
    track_paths,
)
from passagewatch.ingestion.metadata import ClipMetadata

PAGE = 200
OBSERVATION_FRAMES = 500  # the API's limit per request


def frames_zip(frame_dir: Path, num_frames: int) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:  # JPEGs: no gain
        for index in range(num_frames):
            path = frame_dir / f"{index}.jpg"
            if not path.is_file():
                raise FileNotFoundError(f"missing frame {path}")
            archive.write(path, f"{index}.jpg")
    return buffer.getvalue()


def upload(client: httpx.Client, frame_dir: Path, meta: ClipMetadata) -> str:
    response = client.post(
        "/v1/clips",
        files={"file": (f"{meta.clip_name}.zip", frames_zip(frame_dir, meta.num_frames))},
        data={
            "x_meter_start": str(meta.x_meter_start),
            "x_meter_stop": str(meta.x_meter_stop),
            "y_meter_start": str(meta.y_meter_start),
            "y_meter_stop": str(meta.y_meter_stop),
            "framerate": str(meta.framerate),
        },
    )
    response.raise_for_status()
    return str(response.json()["clip_id"])


def create_job(client: httpx.Client, clip_id: str, key: str) -> str:
    response = client.post("/v1/jobs", json={"clip_id": clip_id}, headers={"Idempotency-Key": key})
    response.raise_for_status()
    return str(response.json()["job_id"])


def wait(client: httpx.Client, job_id: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] == "succeeded":
            results: dict[str, Any] = client.get(f"/v1/jobs/{job_id}/results").json()
            return results
        if job["status"] in ("failed", "cancelled"):
            raise RuntimeError(f"job {job_id} {job['status']}: {job.get('error')}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id} still {job['status']} after {timeout_s:.0f} s")
        time.sleep(2)


def fetch(client: httpx.Client, job_id: str) -> tuple[list[TrackLabel], tuple[int, int], int]:
    """A job's track labels (latest revision), reviewed counts, and passages added."""
    tracks: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            client.get(f"/v1/jobs/{job_id}/tracks", params={"offset": offset, "limit": PAGE})
            .raise_for_status()
            .json()
        )
        tracks += page["tracks"]
        offset += PAGE
        if offset >= page["total"]:
            break
    results = client.get(f"/v1/jobs/{job_id}/results").raise_for_status().json()
    counts = results["reviewed"] or results["automatic"]
    report = client.get(f"/v1/jobs/{job_id}/export", params={"format": "json"}).json()
    added = sum(p["state"] == "added" for p in report.get("added_passages", []))
    return track_labels(tracks), (counts["right"], counts["left"]), added


def changes(client: httpx.Client, job_id: str, labels: list[TrackLabel]) -> list[dict[str, Any]]:
    """The changed tracks of a job, described from its observations."""
    if not any(label.verdict in ("removed", "counted", "redirected") for label in labels):
        return []
    job = client.get(f"/v1/jobs/{job_id}").raise_for_status().json()
    clip = client.get(f"/v1/clips/{job['clip_id']}").raise_for_status().json()
    boxes: list[dict[str, Any]] = []
    for start in range(0, clip["num_frames"], OBSERVATION_FRAMES):
        stop = min(clip["num_frames"], start + OBSERVATION_FRAMES)
        params = {"start": start, "stop": stop}
        page = client.get(f"/v1/jobs/{job_id}/observations", params=params)
        boxes += page.raise_for_status().json()["boxes"]
    paths = track_paths(boxes, clip["width"])
    return describe_changes(labels, paths, job["counting"]["line_x_normalized"])


def review_events(client: httpx.Client, job_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = (
        client.get(f"/v1/jobs/{job_id}/reviews").raise_for_status().json()["events"]
    )
    return events
