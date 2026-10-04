"""Release manifests, deployment checks and the rollback procedure (Stage 11)."""

from __future__ import annotations

import io
import threading
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from passagewatch.service.api.app import create_app
from passagewatch.service.bundle import activate, build_bundle, load_bundle
from passagewatch.service.release import (
    ManifestExistsError,
    ReleaseManifest,
    bundle_identity,
    check_model_info,
    read_manifest,
    write_manifest,
)
from passagewatch.service.settings import ServiceSettings
from passagewatch.service.worker.loop import run_worker
from passagewatch.service.worker.pipeline import InferencePipeline

from .test_worker import checkpoint

METERS = {"x_meter_start": -1.0, "x_meter_stop": 1.0, "y_meter_start": 3.0, "y_meter_stop": 0.0}


def manifest_for(bundle_dir: Path) -> ReleaseManifest:
    bundle = load_bundle(bundle_dir)
    return ReleaseManifest(
        release=bundle.pipeline_version,
        created_at="2026-10-04T00:00:00+00:00",
        code_commit="0" * 40,
        uv_lock_sha256="1" * 64,
        dataset_manifest={"version": "full-v2"},
        training={"run": "test"},
        bundle=bundle_identity(bundle),
        evaluation={"reports": []},
    )


def clip_zip(seed: int) -> bytes:
    """A distinct recording per seed, so no job is answered from the result cache."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i in range(10):
            frame = np.full((60, 40), 40 + seed, dtype=np.uint8)
            zf.writestr(f"{i}.png", cv2.imencode(".png", frame)[1].tobytes())
    return buffer.getvalue()


@pytest.fixture
def bundles(tmp_path: Path) -> Path:
    for version, threshold in (("pw-a", 0.3), ("pw-b", 0.5)):
        build_bundle(
            checkpoint=checkpoint(tmp_path / f"{version}.pt"),
            tracker_config={},
            score_threshold=threshold,
            version=version,
            bundles_dir=tmp_path / "bundles",
        )
    return tmp_path / "bundles"


def test_manifests_are_immutable_except_for_filling_in_the_image(
    bundles: Path, tmp_path: Path
) -> None:
    manifest = manifest_for(bundles / "pw-a")
    path = tmp_path / "releases" / "pw-a.json"
    write_manifest(manifest, path)

    write_manifest(manifest.model_copy(update={"created_at": "later"}), path)  # unchanged
    assert read_manifest(path).created_at == manifest.created_at
    write_manifest(manifest.model_copy(update={"image_digest": "sha256:abc"}), path)
    assert read_manifest(path).image_digest == "sha256:abc"
    with pytest.raises(ManifestExistsError, match="already records image"):
        write_manifest(manifest.model_copy(update={"image_digest": "sha256:def"}), path)
    with pytest.raises(ManifestExistsError, match="different content"):
        write_manifest(manifest.model_copy(update={"code_commit": "9" * 40}), path)


def deploy(
    bundles: Path, tmp_path: Path, version: str
) -> tuple[ServiceSettings, InferencePipeline]:
    """What a deployment does: point bundles/active at a release, restart API and worker."""
    activate(bundles, version)
    settings = ServiceSettings(
        data_dir=tmp_path / "var",  # the same database and artifacts across deployments
        bundle_dir=(bundles / "active").resolve(),
        worker_poll_seconds=0.05,
        heartbeat_seconds=0.2,
        lease_seconds=5,
    )
    return settings, InferencePipeline.load(settings.bundle_dir, threads=1)


def run_job(
    settings: ServiceSettings, pipeline: InferencePipeline, seed: int
) -> tuple[str, dict[str, Any]]:
    with TestClient(create_app(settings)) as client:
        clip = client.post(
            "/v1/clips",
            files={"file": ("c.zip", clip_zip(seed))},
            data={k: str(v) for k, v in (METERS | {"framerate": 10}).items()},
        ).json()
        job_id = client.post("/v1/jobs", json={"clip_id": clip["clip_id"]}).json()["job_id"]
        run_worker(settings, pipeline, "w", stop=threading.Event(), max_jobs=1)
        return job_id, client.get("/v1/model-info").json()


def test_rollback_restores_the_previous_release_and_old_jobs_keep_theirs(
    bundles: Path, tmp_path: Path
) -> None:
    a, b = manifest_for(bundles / "pw-a"), manifest_for(bundles / "pw-b")

    job_a, info = run_job(*deploy(bundles, tmp_path, "pw-a"), seed=1)
    assert check_model_info(a, info) == []
    job_b, info = run_job(*deploy(bundles, tmp_path, "pw-b"), seed=2)
    assert check_model_info(b, info) == []
    assert any("score_threshold" in p for p in check_model_info(a, info))

    settings, pipeline = deploy(bundles, tmp_path, "pw-a")  # roll back
    job_after, info = run_job(settings, pipeline, seed=3)
    assert check_model_info(a, info) == []
    with TestClient(create_app(settings)) as client:
        versions = {
            j: client.get(f"/v1/jobs/{j}").json()["pipeline_version"]
            for j in (job_a, job_b, job_after)
        }
        export_b = client.get(f"/v1/jobs/{job_b}/export").json()
    assert versions == {job_a: "pw-a", job_b: "pw-b", job_after: "pw-a"}
    assert export_b["pipeline"]["pipeline_version"] == "pw-b"
    assert export_b["pipeline"]["detector"]["score_threshold"] == 0.5


def test_the_release_gate_requires_parity_and_a_merged_code_commit(
    bundles: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_release", Path(__file__).resolve().parents[2] / "scripts/check_release.py"
    )
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)  # not a git repository: commit not found
    manifest = manifest_for(bundles / "pw-a").model_copy(
        update={
            "evaluation": {
                "reports": [
                    {"partition": "val", "runtime": "torch", "nmae": 0.1, "nmae_95ci": [0, 1]}
                ],
                "parity": [{"report": "r-onnx.json", "clips_with_different_counts": 2}],
            }
        }
    )
    write_manifest(manifest, tmp_path / "releases/manifests/pw-a.json")

    assert gate.main(["--release", "pw-a"]) == 1
    out = capsys.readouterr().out
    assert "parity with r-onnx.json: 2 clips differ" in out
    assert "is not in this history" in out
    assert gate.main(["--release", "pw-missing"]) == 1
