"""The same-fish re-test's analysis reads reviewers' decisions and answers from the API
(scripts/analyze_retest.py; docs/review_retest.md)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from fastapi.testclient import TestClient

from passagewatch.evaluation.same_fish import SAME_FISH_REASONS

from .test_reviews import client, finished_job, review, settings  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]


def analysis() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_retest", REPO_ROOT / "scripts/analyze_retest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_analysis_reads_answers_errors_and_who_started(client: TestClient, settings) -> None:  # noqa: F811
    reviewed = finished_job(client, settings)  # tracks 1, 2 right; 3 left; 4 not counted
    clip_id = client.get(f"/v1/jobs/{reviewed}").json()["clip_id"]
    untouched = client.post(  # another reviewer's job: a result-cache hit, as in the re-test
        "/v1/jobs", json={"clip_id": clip_id}, headers={"Idempotency-Key": "retest-T2-t01"}
    ).json()["job_id"]
    assert client.get(f"/v1/jobs/{untouched}").json()["status"] == "succeeded"
    for revision, track in enumerate((1, 2, 3)):
        review(client, reviewed, base_revision=revision, action="accept", track_id=track)
    different = SAME_FISH_REASONS["different"]
    review(
        client, reviewed, base_revision=3, action="set_direction", track_id=4,
        direction="right", reason=different,
    )  # fmt: skip
    plan = {
        "design": "docs/review_retest.md",
        "release": "test",
        "review_page_commit": "abc",
        "clips": {
            "t01": {"set": "A", "sample_code": "s01", "reference": [2, 1], "automatic": [2, 1]}
        },
        "reviewers": ["T1", "T2"],
        "jobs": {"T1": {"t01": reviewed}, "T2": {"t01": untouched}},
    }

    result = analysis().analyze(client, plan, REPO_ROOT / "review_sample/results.json")

    t1 = result["reviewers"]["T1"]
    assert t1["clips"]["t01"]["answers"] == {"same": 0, "different": 1}
    assert t1["clips"]["t01"]["complete"] and t1["clips"]["t01"]["reviewed"] == [3, 1]
    assert t1["all"]["automatic_error"] == 0 and t1["all"]["reviewed_error"] == 1
    assert t1["target_no_harm"] is False
    assert result["not_started"] == ["T2"]
    assert result["targets"]["no_harm"]["outcome"].startswith("not evaluable")
    # The design's historical figures on set A (docs/review_retest.md).
    history = result["historical_set_a"]
    assert history["R1"]["fragment_counts"] == 5 and history["R2"]["fragment_counts"] == 15
    assert history["R1"]["reviewed_error"] == 11 and history["R2"]["reviewed_error"] == 18
