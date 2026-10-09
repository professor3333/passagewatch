"""Demo examples in the service (docs/demos.md): listing, retention, deletion, visitor copies."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from passagewatch.service.api.app import create_app
from passagewatch.service.demos import DemoCatalog, catalog_json
from passagewatch.service.settings import ServiceSettings

from .fakes import passage_zip
from .test_worker import METERS, make_settings, work

DEMO_ZIP = passage_zip(20)


def catalog(sha256: str) -> DemoCatalog:
    return DemoCatalog.model_validate(
        {
            "version": "demos-v1",
            "demos": [
                {
                    "demo_id": "clear",
                    "kind": "clear",
                    "title": "A clear passage",
                    "summary": "One target crossing left to right.",
                    "source": {
                        "dataset": "Caltech Fish Counting (CFC) v1.1",
                        "location": "kenai-train",
                        "clip_name": "synthetic",
                        "split_note": "test fixture",
                    },
                    "reference": {"right": 1, "left": 0},
                    "framerate": 10,
                    "meters": {"x_start": -1.0, "x_stop": 1.0, "y_start": 3.0, "y_stop": 0.0},
                    "num_frames": 20,
                    "sha256": sha256,
                }
            ],
        }
    )


@pytest.fixture
def settings(tmp_path: Path) -> ServiceSettings:
    path = tmp_path / "catalog.json"
    path.write_text(catalog_json(catalog(hashlib.sha256(DEMO_ZIP).hexdigest())))
    return make_settings(tmp_path, demo_catalog=path)


def upload(client: TestClient, data: bytes) -> dict[str, Any]:
    response = client.post(
        "/v1/clips",
        files={"file": ("clip.zip", data)},
        data={k: str(v) for k, v in (METERS | {"framerate": 10}).items()},
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


def test_a_demo_is_listed_once_its_precomputed_job_succeeds(settings: ServiceSettings) -> None:
    with TestClient(create_app(settings)) as client:
        (demo,) = client.get("/v1/demos").json()["demos"]
        assert (demo["demo_id"], demo["available"], demo["job_id"]) == ("clear", False, None)

        clip = upload(client, DEMO_ZIP)
        job = client.post("/v1/jobs", json={"clip_id": clip["clip_id"]}).json()["job_id"]
    assert work(settings) == 1

    with TestClient(create_app(settings)) as client:
        listing = client.get("/v1/demos").json()
        (demo,) = listing["demos"]
        assert listing["version"] == "demos-v1"
        assert demo["available"] and demo["clip_id"] == clip["clip_id"]
        assert demo["job_id"] == job and demo["computed_at"] is not None
        assert demo["reference"] == {"right": 1, "left": 0}

        # A visitor opens the demo as their own job: a cache hit, labelled as such, and the
        # listing keeps pointing at the original analysis.
        visitor = client.post(
            "/v1/jobs", json={"clip_id": clip["clip_id"]}, headers={"Idempotency-Key": "v-1"}
        ).json()["job_id"]
        copy = client.get(f"/v1/jobs/{visitor}").json()
        assert copy["status"] == "succeeded" and copy["cached_from"] == job
        assert client.get(f"/v1/jobs/{visitor}/results").json()["cached"]
        assert client.get("/v1/demos").json()["demos"][0]["job_id"] == job


def test_demo_recordings_never_expire_and_cannot_be_deleted(settings: ServiceSettings) -> None:
    with TestClient(create_app(settings)) as client:
        demo = upload(client, DEMO_ZIP)
        other = upload(client, passage_zip(21))

        assert demo["expires_at"] is None
        assert other["expires_at"] is not None
        assert client.delete(f"/v1/clips/{demo['clip_id']}").status_code == 403
        assert client.delete(f"/v1/clips/{other['clip_id']}").status_code == 204


def test_without_a_catalog_there_are_no_demos(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        assert client.get("/v1/demos").json() == {"version": None, "demos": []}
        assert upload(client, DEMO_ZIP)["expires_at"] is not None
