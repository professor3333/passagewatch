"""Prometheus metrics of the API and the worker (Stage 11)."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi.testclient import TestClient
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from passagewatch.service.api.app import create_app
from passagewatch.service.metrics import WorkerMetrics
from passagewatch.service.worker.loop import run_worker

from .fakes import passage_zip, pipeline
from .test_worker import make_settings, submit


def samples(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    return {
        (s.name, tuple(sorted(s.labels.items()))): s.value
        for family in text_string_to_metric_families(text)
        for s in family.samples
    }


def test_metrics_report_jobs_queue_workers_and_the_release(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    worker_metrics = WorkerMetrics()
    with TestClient(create_app(settings)) as client:
        before = samples(client.get("/metrics").text)
        assert before[("passagewatch_live_workers", ())] == 0
        submit(client, passage_zip())
        queued = samples(client.get("/metrics").text)
        assert queued[("passagewatch_queue_depth", ())] == 1
        assert queued[("passagewatch_jobs", (("status", "queued"),))] == 1

        assert (
            run_worker(
                settings,
                pipeline(),
                "w",
                stop=threading.Event(),
                max_jobs=1,
                metrics=worker_metrics,
            )
            == 1
        )
        response = client.get("/metrics")

    after = samples(response.text)
    assert response.headers["content-type"].startswith("text/plain")
    assert after[("passagewatch_jobs", (("status", "succeeded"),))] == 1
    assert after[("passagewatch_queue_depth", ())] == 0
    assert after[("passagewatch_oldest_queued_seconds", ())] == 0
    assert after[("passagewatch_live_workers", ())] == 1
    info = [k for k in after if k[0] == "passagewatch_release_info"]
    assert info and ("pipeline_version", "pw-test-1") in info[0][1]

    worker = samples(generate_latest(worker_metrics.registry).decode())
    assert worker[("passagewatch_worker_jobs_total", (("outcome", "succeeded"),))] == 1
    assert worker[("passagewatch_worker_frames_total", ())] == 20
    assert worker[("passagewatch_worker_job_seconds_count", ())] == 1
    assert ("passagewatch_worker_stage_seconds_total", (("stage", "detect"),)) in worker
