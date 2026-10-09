"""The review sample's analysis reads a reviewer's decisions from the API
(scripts/analyze_review_sample.py; docs/review_sample.md)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from fastapi.testclient import TestClient

from .test_reviews import client, finished_job, review, settings  # noqa: F401

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_review_sample.py"


def analysis() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyze_review_sample", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_reads_verdicts_counts_and_added_passages(client: TestClient, settings) -> None:  # noqa: F811
    job_id = finished_job(client, settings)  # tracks 1, 2 right; 3 left; 4 not counted
    assert review(client, job_id, base_revision=0, action="accept", track_id=1).status_code == 201
    review(client, job_id, base_revision=1, action="reject", track_id=2, reason="debris")
    review(client, job_id, base_revision=2, action="set_direction", track_id=4, direction="left")
    review(client, job_id, base_revision=3, action="mark_unresolved", track_id=3)
    review(client, job_id, base_revision=4, action="add_passage", direction="right", frame_index=3)

    labels, counts, added = analysis().fetch(client, job_id)

    verdicts = {label.track_id: label.verdict for label in labels}
    assert verdicts == {1: "kept", 2: "removed", 3: "unresolved", 4: "counted"}
    assert counts == (2, 1)  # track 1 and the added fish right; track 4 left
    assert added == 1
