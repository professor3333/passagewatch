"""The usability study's endpoints (docs/usability_study.md)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from passagewatch.service.api.app import create_app
from passagewatch.service.settings import ServiceSettings
from passagewatch.service.worker.loop import run_worker

from .fakes import passage_zip, pipeline
from .test_worker import make_settings, submit


def trial(**overrides: Any) -> dict[str, Any]:
    body = {
        "participant": "D1",
        "block": 1,
        "condition": "manual",
        "clip": "c01",
        "started_at": "2026-10-05T10:00:00+00:00",
        "finished_at": "2026-10-05T10:01:30+00:00",
        "active_ms": 90_000,
        "pauses": 0,
        "right": 1,
        "left": 0,
    }
    return body | overrides


@pytest.fixture
def study(tmp_path: Path) -> ServiceSettings:
    """Four analysed clips (two practice, S1, S2) and a plan for two participants.

    D1 counts S1 by hand and reviews S2; P1 reviews S1 and counts S2. Each assisted trial has
    its own job (a result-cache hit), as ``scripts/prepare_usability_study.py`` creates them.
    """
    base = make_settings(tmp_path)
    keys = ["p01", "p02", "c01", "c02"]
    with TestClient(create_app(base)) as client:
        first = {
            k: submit(client, passage_zip(n)) for k, n in zip(keys, (20, 21, 22, 23), strict=True)
        }
        clip_ids = {k: client.get(f"/v1/jobs/{j}").json()["clip_id"] for k, j in first.items()}
    run_worker(base, pipeline(), "w", stop=threading.Event(), max_jobs=4)

    def job(participant: str, key: str) -> str:
        with TestClient(create_app(base)) as client:
            response = client.post(
                "/v1/jobs",
                json={"clip_id": clip_ids[key]},
                headers={"Idempotency-Key": f"study-{participant}-{key}"},
            )
        assert response.status_code == 202
        return str(response.json()["job_id"])

    def clip(key: str, role: str) -> dict[str, Any]:
        return {
            "clip_name": f"clip-{key}",
            "clip_id": clip_ids[key],
            "role": role,
            "camera": "LeftFar",
            "num_frames": 20,
            "framerate": 10.0,
            "reference": [1, 0],
            "automatic": [1, 0],
        }

    plan = {
        "design": "docs/usability_study.md",
        "release": "pw-test-1",
        "seed": 0,
        "clips": {
            "p01": clip("p01", "practice"),
            "p02": clip("p02", "practice"),
            "c01": clip("c01", "S1"),
            "c02": clip("c02", "S2"),
        },
        "practice": ["p01", "p02"],
        "sets": {"S1": ["c01"], "S2": ["c02"]},
        "participants": {
            "D1": [{"condition": "manual", "set": "S1"}, {"condition": "assisted", "set": "S2"}],
            "P1": [{"condition": "assisted", "set": "S1"}, {"condition": "manual", "set": "S2"}],
        },
        "jobs": {
            "D1": {"p02": job("D1", "p02"), "c02": job("D1", "c02")},
            "P1": {"p01": job("P1", "p01"), "c01": job("P1", "c01")},
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return base.model_copy(update={"study_mode": True, "study_plan": path})


def test_the_plan_hides_the_answers(study: ServiceSettings) -> None:
    with TestClient(create_app(study)) as client:
        plan = client.get("/v1/study/plan").json()

    assert set(plan["clips"]["c01"]) == {"clip_id", "role", "num_frames", "framerate"}
    assert plan["participants"]["D1"][0] == {"condition": "manual", "set": "S1"}
    assert set(plan["jobs"]["P1"]) == {"p01", "c01"}


def test_trials_record_manual_counts_and_assisted_review_revisions(study: ServiceSettings) -> None:
    with TestClient(create_app(study)) as client:
        manual = client.post("/v1/study/trials", json=trial(right=2, left=1))
        job_id = client.get("/v1/study/plan").json()["jobs"]["D1"]["c02"]
        review = client.post(
            f"/v1/jobs/{job_id}/reviews",
            json={
                "base_revision": 0,
                "action": "add_passage",
                "direction": "left",
                "frame_index": 3,
            },
        )
        assisted = client.post(
            "/v1/study/trials",
            json=trial(block=2, condition="assisted", clip="c02", right=None, left=None),
        )
        practice = client.post("/v1/study/trials", json=trial(clip="p01", right=1, left=0))
        csv = client.get("/v1/study/export/trials.csv").text
        progress = client.get("/v1/study/progress/D1").json()

    assert manual.status_code == 201 and manual.json()["final"] == [2, 1]
    assert review.status_code == 201
    assert assisted.status_code == 201
    assert assisted.json() == {"clip": "c02", "final": [1, 1], "revision": 1}  # the reviewed counts
    assert practice.status_code == 201
    lines = csv.strip().splitlines()
    assert lines[0].startswith("participant,block,condition,clip,practice")
    assert len(lines) == 4
    assert sorted(progress["clips_done"]) == ["c01", "c02", "p01"]


def test_trials_outside_the_plan_or_repeated_are_refused(study: ServiceSettings) -> None:
    with TestClient(create_app(study)) as client:
        wrong_set = client.post("/v1/study/trials", json=trial(clip="c02"))
        other_practice = client.post("/v1/study/trials", json=trial(clip="p02"))
        wrong_condition = client.post("/v1/study/trials", json=trial(condition="assisted"))
        unknown = client.post("/v1/study/trials", json=trial(participant="X9"))
        no_counts = client.post("/v1/study/trials", json=trial(right=None))
        first = client.post("/v1/study/trials", json=trial())
        again = client.post("/v1/study/trials", json=trial())

    refused = (wrong_set, other_practice, wrong_condition, unknown, no_counts)
    assert [r.status_code for r in refused] == [422] * 5
    assert first.status_code == 201 and again.status_code == 409


def test_each_participant_reviews_their_own_copy(study: ServiceSettings) -> None:
    with TestClient(create_app(study)) as client:
        jobs = client.get("/v1/study/plan").json()["jobs"]
        d1, p1 = jobs["D1"]["c02"], jobs["P1"]["c01"]
        assert len({d1, p1, jobs["D1"]["p02"], jobs["P1"]["p01"]}) == 4
        client.post(
            f"/v1/jobs/{d1}/reviews",
            json={
                "base_revision": 0,
                "action": "add_passage",
                "direction": "left",
                "frame_index": 3,
            },
        )
        p1_trial = client.post(
            "/v1/study/trials",
            json=trial(participant="P1", condition="assisted", right=None, left=None),
        )

    assert p1_trial.json() == {"clip": "c01", "final": [1, 0], "revision": 0}  # untouched


def test_questionnaires_are_validated(study: ServiceSettings) -> None:
    form = {"participant": "D1", "condition": "manual", "sus": [3] * 10, "tlx": [50] * 6}
    with TestClient(create_app(study)) as client:
        ok = client.post("/v1/study/questionnaires", json=form)
        bad_sus = client.post("/v1/study/questionnaires", json=form | {"sus": [6] * 10})
        bad_tlx = client.post("/v1/study/questionnaires", json=form | {"tlx": [50] * 5})
        csv = client.get("/v1/study/export/questionnaires.csv").text

    assert ok.status_code == 201
    assert bad_sus.status_code == bad_tlx.status_code == 422
    assert "D1,manual" in csv


def test_study_endpoints_exist_only_in_study_mode(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        assert client.get("/v1/study/plan").status_code == 404


def test_the_service_runs_before_the_plan_is_prepared(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, study_mode=True, study_plan=tmp_path / "missing.json")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/v1/study/plan").status_code == 503
