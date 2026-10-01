"""The deployed pipeline must count exactly like the offline evaluation it was selected by.

A real kenai-val clip is uploaded through the API and run by the worker with the active
release bundle (yolox-tiny-v1 epoch 25, score threshold 0.2, classical-v2 tracker). Its
counts must equal those that ``evaluate_neural.py`` recorded for the same clip, epoch and
threshold. Filtering by score before NMS (service) and after it (offline cache) keeps the
same boxes, because greedy NMS only lets higher-scoring boxes suppress lower-scoring ones.
"""

from __future__ import annotations

import io
import json
import threading
import zipfile
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from passagewatch.ingestion.cfc import CfcLayout
from passagewatch.service.api.app import create_app
from passagewatch.service.bundle import load_bundle
from passagewatch.service.settings import ServiceSettings
from passagewatch.service.worker.loop import run_worker
from passagewatch.service.worker.pipeline import InferencePipeline

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "bundles/active"
FRAMES = REPO / "data/extracted/cfc/kenai-dev-v1"
REPORT = REPO / "runs/neural/yolox-tiny-v1/report-val.json"


@pytest.mark.slow
@pytest.mark.skipif(
    not (BUNDLE.exists() and FRAMES.is_dir() and REPORT.is_file()),
    reason="needs the active bundle, kenai-dev-v1 frames and the offline report",
)
def test_service_counts_equal_the_offline_evaluation(tmp_path: Path) -> None:
    bundle = load_bundle(BUNDLE.resolve())
    report = json.loads(REPORT.read_text())
    offline = next(
        r
        for r in report["results"]
        if r["epoch"] == bundle.detector.epoch and r["threshold"] == bundle.detector.score_threshold
    )
    assert offline["checkpoint_sha256"] == bundle.detector.checkpoint_sha256
    layout = CfcLayout.full(REPO / "data/extracted/cfc")
    meta_by_clip = layout.metadata("kenai-val").clips
    # The smallest-frame clips with at least one passage, for speed.
    candidates = [
        c
        for c in offline["clips"]
        if sum(c["reference"]) > 0 and meta_by_clip[c["clip_name"]].width < 300
    ]
    chosen = min(candidates, key=lambda c: meta_by_clip[c["clip_name"]].num_frames)
    meta = meta_by_clip[chosen["clip_name"]]

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        for i in range(meta.num_frames):  # original JPEG bytes: identical decoding
            zf.writestr(
                f"{i}.jpg", (FRAMES / "kenai-val" / meta.clip_name / f"{i}.jpg").read_bytes()
            )
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # offline ran on MPS
    settings = ServiceSettings(
        data_dir=tmp_path / "var", bundle_dir=BUNDLE.resolve(), max_frames=10_000, device=device
    )

    with TestClient(create_app(settings)) as client:
        clip = client.post(
            "/v1/clips",
            files={"file": ("clip.zip", archive.getvalue())},
            data={
                "framerate": str(meta.framerate),
                "x_meter_start": str(meta.x_meter_start),
                "x_meter_stop": str(meta.x_meter_stop),
                "y_meter_start": str(meta.y_meter_start),
                "y_meter_stop": str(meta.y_meter_stop),
            },
        ).json()
        job_id = client.post("/v1/jobs", json={"clip_id": clip["clip_id"]}).json()["job_id"]
        pipeline = InferencePipeline.load(BUNDLE.resolve(), device)
        run_worker(settings, pipeline, "w-regression", stop=threading.Event(), max_jobs=1)
        results = client.get(f"/v1/jobs/{job_id}/results").json()

    automatic = results["automatic"]
    assert [automatic["right"], automatic["left"]] == chosen["predicted"]
