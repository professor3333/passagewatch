"""Verify that a running deployment is the release it claims to be, and that it works.

1. ``/health/ready`` answers 200: the API is up and a worker with the release loaded is alive.
2. ``/v1/model-info`` matches the release manifest field by field: pipeline version and
   config hash, checkpoint and ONNX hashes, input size, threshold, preprocessing version,
   tracker configuration, counting-policy and calibration versions.
3. An example job (a generated clip with one target crossing the line) is uploaded, runs and
   succeeds; its counts are reported, not judged, since the example is synthetic.

Exits non-zero on any failure.

Example:
    uv run python scripts/verify_deployment.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from passagewatch.service.release import check_model_info, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def request(
    method: str, url: str, body: bytes | None = None, headers: dict[str, str] | None = None
) -> Any:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read())


def example_clip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i in range(30):
            frame = np.full((200, 120), 40, dtype=np.uint8)
            frame[100:108, 6 + 3 * i : 20 + 3 * i] = 230
            zf.writestr(f"{i}.png", cv2.imencode(".png", frame)[1].tobytes())
    return buffer.getvalue()


def multipart(fields: dict[str, str], file_bytes: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        header = f'Content-Disposition: form-data; name="{key}"'
        parts.append(f"--{boundary}\r\n{header}\r\n\r\n{value}\r\n".encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="example.zip"\r\n'
        "Content-Type: application/zip\r\n\r\n".encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", type=Path, default=None, help="default: the served release's")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds for the example job")
    args = parser.parse_args(argv)
    url = args.url.rstrip("/")
    failures: list[str] = []

    try:
        request("GET", f"{url}/health/ready")
        print("ready: yes")
    except Exception as exc:  # any failure to reach readiness is reported
        print(f"ready: NO ({exc})")
        return 1

    info = request("GET", f"{url}/v1/model-info")
    manifest_path = (
        args.manifest or REPO_ROOT / "releases/manifests" / f"{info['pipeline_version']}.json"
    )
    if not manifest_path.is_file():
        print(f"release manifest {manifest_path} not found")
        return 1
    manifest = read_manifest(manifest_path)
    problems = check_model_info(manifest, info)
    if problems:
        failures.extend(f"model-info: {p}" for p in problems)
    print(
        f"release: {info['pipeline_version']} vs manifest {manifest.release}: "
        f"{'match' if not problems else f'{len(problems)} difference(s)'}"
    )

    body, content_type = multipart(
        {
            "framerate": "10",
            "x_meter_start": "-1",
            "x_meter_stop": "1",
            "y_meter_start": "3",
            "y_meter_stop": "0",
        },
        example_clip(),
    )
    clip = request("POST", f"{url}/v1/clips", body, {"Content-Type": content_type})
    job = request(
        "POST",
        f"{url}/v1/jobs",
        json.dumps({"clip_id": clip["clip_id"]}).encode(),
        {"Content-Type": "application/json", "Idempotency-Key": f"verify-{uuid.uuid4().hex}"},
    )
    deadline = time.monotonic() + args.timeout
    status = "queued"
    while time.monotonic() < deadline:
        status = request("GET", f"{url}/v1/jobs/{job['job_id']}")["status"]
        if status in ("succeeded", "failed"):
            break
        time.sleep(2)
    if status != "succeeded":
        failures.append(f"example job {job['job_id']}: {status}")
    else:
        results = request("GET", f"{url}/v1/jobs/{job['job_id']}/results")
        print(
            f"example job: succeeded with {results['pipeline_version']} "
            f"(right {results['automatic']['right']}, left {results['automatic']['left']})"
        )
        if results["pipeline_version"] != manifest.release:
            failures.append(f"example job ran {results['pipeline_version']}")

    for failure in failures:
        print(f"FAIL {failure}")
    print("deployment verified" if not failures else "deployment NOT verified")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
