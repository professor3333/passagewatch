from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from passagewatch.detection.classical import FrameDetections
from passagewatch.detection.suppression import SuppressionConfig
from passagewatch.evaluation.ablation import evaluate_variant, plan_variants, select
from passagewatch.ingestion.cfc import Clip
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.mot import BoxAnnotations
from passagewatch.tracking.kalman import TrackerConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = {"tracking_config": "configs/tracking/classical-v2.yaml", "score_threshold": 0.2}


def test_variants_override_only_what_they_change() -> None:
    plan = {
        "base": BASE,
        "variants": {
            "current": {},
            "strict": {"suppression": {"containment": 0.8}},
            "gate": {"tracker": {"gate_m": 0.3}, "score_threshold": 0.3},
        },
    }

    current, strict, gate = plan_variants(plan, REPO_ROOT)

    assert current.suppression == SuppressionConfig() and current.score_threshold == 0.2
    assert strict.suppression.containment == 0.8 and strict.tracker == current.tracker
    assert gate.tracker.gate_m == 0.3 and gate.tracker.max_age == current.tracker.max_age
    assert gate.score_threshold == 0.3


def test_a_plan_must_start_with_the_current_pipeline_and_use_known_keys() -> None:
    with pytest.raises(ValueError, match="current"):
        plan_variants({"base": BASE, "variants": {"other": {}}}, REPO_ROOT)
    with pytest.raises(ValueError, match="unknown keys"):
        plan_variants({"base": BASE, "variants": {"current": {"nms": 0.5}}}, REPO_ROOT)


def crossing_clip() -> tuple[Clip, list[FrameDetections]]:
    """One fish crossing left to right; the detector also boxes its tail half."""
    meta = ClipMetadata(
        clip_name="c_2018-06-01_120000_0_30",
        num_frames=30,
        framerate=10.0,
        width=400,
        height=100,
        x_meter_start=0.0,
        x_meter_stop=4.0,
        y_meter_start=1.0,
        y_meter_stop=0.0,
    )
    frames = np.arange(30)
    x = 20.0 + 12.0 * frames
    whole = np.stack([x, np.full(30, 40.0), x + 40, np.full(30, 56.0)], axis=1)
    reference = BoxAnnotations(frames, np.ones(30, dtype=np.int64), whole)
    clip = Clip("kenai-val", meta, 0, 30, reference, None)
    # The tail box lies inside the fish box (IoU 0.375, containment 1.0); with one-to-one
    # association, the tracker gives each box its own track.
    tail = whole + np.array([0.0, 2.0, -20.0, -2.0])
    detections = [
        FrameDetections(np.stack([whole[i], tail[i]]), np.array([0.9, 0.6])) for i in range(30)
    ]
    return clip, detections


def test_containment_suppression_removes_the_double_count() -> None:
    clip, detections = crossing_clip()
    plan = {
        "base": BASE,
        "variants": {"current": {}, "contain": {"suppression": {"containment": 0.8}}},
    }
    current, contain = plan_variants(plan, REPO_ROOT)

    before = evaluate_variant([clip], {clip.name: detections}, current)
    after = evaluate_variant([clip], {clip.name: detections}, contain)

    assert before.errors[0].predicted.right == 2 and before.passage_errors == 1
    assert before.passages["predicted_passages"]["duplicate"] == 1
    assert after.errors[0].predicted.right == 1 and after.nmae == 0.0
    assert after.passage_errors == 0
    assert select([before, after]) is after


def test_select_keeps_the_current_pipeline_on_a_tie() -> None:
    clip, detections = crossing_clip()
    plan = {
        "base": BASE,
        "variants": {"current": {}, "same": {"tracker": TrackerConfig().model_dump()}},
    }
    variants = plan_variants(plan, REPO_ROOT)
    results = [evaluate_variant([clip], {clip.name: detections}, v) for v in variants]
    tied = [results[0], results[0].__class__(variants[1], results[0].errors, results[0].passages)]

    assert select(list(reversed(tied))).variant.name == "current"
