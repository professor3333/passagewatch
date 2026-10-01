"""Test doubles for the worker: a fast detector for synthetic frames, and a slow variant.

Importable by name (``integration.fakes``) so that a spawned worker process can use them.
"""

from __future__ import annotations

import io
import time
import zipfile
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from passagewatch.detection.classical import FrameDetections
from passagewatch.service.bundle import ReleaseBundle
from passagewatch.service.worker.pipeline import InferencePipeline

BUNDLE: dict[str, Any] = {
    "pipeline_version": "pw-test-1",
    "detector": {
        "model_size": "tiny",
        "checkpoint": "detector.pt",
        "checkpoint_sha256": "d" * 64,
        "input_height": 960,
        "input_width": 416,
        "score_threshold": 0.2,
        "training_run": "yolox-tiny-v1",
        "epoch": 25,
    },
    "preprocessing_version": "letterbox-gray3-v1",
    "tracker": {"kind": "kalman", "config": {"gate_m": 0.5, "min_length": 3, "min_hits": 3}},
    "counting_policy": "cfc-compatible-v1",
    "provenance": {"note": "test"},
}
H, W = 60, 40  # frames of a 2 m x 3 m window: 5 cm per pixel


class BrightBoxDetector:
    """Finds the bright target that :func:`passage_zip` draws."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def detect(self, frames: list[NDArray[np.uint8]]) -> list[FrameDetections]:
        time.sleep(self.delay)
        out = []
        for frame in frames:
            ys, xs = np.nonzero(frame > 200)
            if len(xs):
                box = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=np.float64)
                out.append(FrameDetections(box, np.array([0.9])))
            else:
                out.append(FrameDetections(np.empty((0, 4)), np.empty(0)))
        return out


def pipeline(delay: float = 0.0) -> InferencePipeline:
    return InferencePipeline(
        ReleaseBundle.model_validate(BUNDLE), BrightBoxDetector(delay), batch_size=4
    )


def passage_zip(n: int = 20, corrupt: int | None = None) -> bytes:
    """A target crossing left to right; frame ``corrupt`` (if given) is not an image."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i in range(n):
            image = np.full((H, W), 40, dtype=np.uint8)
            x = 2 + int(1.5 * i)
            image[30:36, x : x + 8] = 240
            data = b"garbage" if i == corrupt else bytes(cv2.imencode(".png", image)[1])
            zf.writestr(f"{i}.png", data)
    return buffer.getvalue()


def run_slow_worker(settings_json: str, worker_id: str) -> None:
    """Entry point for a worker process that can be killed mid-job."""
    import threading

    from passagewatch.service.settings import ServiceSettings
    from passagewatch.service.worker.loop import run_worker

    settings = ServiceSettings.model_validate_json(settings_json)
    run_worker(settings, pipeline(delay=1.0), worker_id, stop=threading.Event())
