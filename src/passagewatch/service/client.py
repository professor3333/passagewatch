"""Small HTTP helpers for scripts that drive a running PassageWatch service: upload a CFC
clip's frames, create a job, and wait for it (the usability study, the review sample).

Development tooling only: it needs ``httpx``, a development dependency, so the service itself
never imports it."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx

from passagewatch.ingestion.metadata import ClipMetadata


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
