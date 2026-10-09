"""Load the demo examples (docs/demos.md) into a running service.

Downloads ``passagewatch-<version>.tar.gz`` from the GitHub release named in ``--tag`` (or
uses ``--archive``), checks every frames ZIP against ``demos/catalog.json``, and uploads each
demo that the service does not list as available yet. The service's active release analyses
each one once; that job becomes the demo's precomputed (cached) result. The service must
run with the same catalog (``PASSAGEWATCH_DEMO_CATALOG``), so that the uploads are kept.

Example:
    uv run python scripts/load_demos.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import httpx

from passagewatch.service.demos import DemoEntry, load_catalog, unpack_demos

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = "https://github.com/professor3333/passagewatch/releases/download"


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "passagewatch-load-demos"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)


def load(client: httpx.Client, demo: DemoEntry, data: Path, timeout_s: float) -> dict[str, Any]:
    meters = demo.meters
    clip = (
        client.post(
            "/v1/clips",
            files={"file": (demo.zip_name, data.read_bytes())},
            data={
                "x_meter_start": str(meters.x_start),
                "x_meter_stop": str(meters.x_stop),
                "y_meter_start": str(meters.y_start),
                "y_meter_stop": str(meters.y_stop),
                "framerate": str(demo.framerate),
            },
        )
        .raise_for_status()
        .json()
    )
    if clip["expires_at"] is not None:
        raise SystemExit(f"{demo.demo_id}: the service does not treat this upload as a demo")
    job_id = (
        client.post(
            "/v1/jobs",
            json={"clip_id": clip["clip_id"]},
            headers={"Idempotency-Key": f"demo-{demo.demo_id}-{clip['clip_id']}"},
        )
        .raise_for_status()
        .json()["job_id"]
    )
    deadline = time.monotonic() + timeout_s
    while True:
        job = client.get(f"/v1/jobs/{job_id}").raise_for_status().json()
        if job["status"] == "succeeded":
            return dict(client.get(f"/v1/jobs/{job_id}/results").raise_for_status().json())
        if job["status"] == "failed":
            raise SystemExit(f"{demo.demo_id}: job {job_id} failed: {job['error']}")
        if time.monotonic() > deadline:
            raise SystemExit(f"{demo.demo_id}: job {job_id} still {job['status']}")
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "demos/catalog.json")
    parser.add_argument("--tag", default="demos-v1", help="the GitHub release with the archive")
    parser.add_argument("--archive", type=Path, help="use a downloaded archive instead")
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    with (
        tempfile.TemporaryDirectory() as tmp,
        httpx.Client(base_url=args.url, timeout=300) as client,
    ):
        listing = client.get("/v1/demos").raise_for_status().json()
        if listing["version"] != catalog.version:
            raise SystemExit(
                f"the service's demo catalog is {listing['version']}, not {catalog.version}: "
                "set PASSAGEWATCH_DEMO_CATALOG to demos/catalog.json"
            )
        available = {d["demo_id"] for d in listing["demos"] if d["available"]}
        if available == {d.demo_id for d in catalog.demos}:
            print("every demo is already loaded")
            return 0
        archive = args.archive
        if archive is None:
            url = f"{DOWNLOADS}/{args.tag}/{catalog.archive_name}"
            print(f"downloading {url}")
            archive = Path(tmp) / catalog.archive_name
            download(url, archive)
        zips = unpack_demos(archive, catalog, Path(tmp) / "zips")
        for demo in catalog.demos:
            if demo.demo_id in available:
                print(f"{demo.demo_id}: already loaded")
                continue
            result = load(client, demo, zips[demo.demo_id], args.job_timeout)
            auto = result["automatic"]
            print(
                f"{demo.demo_id}: analysed by {result['pipeline_version']}: "
                f"→ {auto['right']} ← {auto['left']} "
                f"(CFC reference → {demo.reference.right} ← {demo.reference.left})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
