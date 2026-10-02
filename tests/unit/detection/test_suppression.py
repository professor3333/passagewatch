from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from passagewatch.detection.classical import FrameDetections
from passagewatch.detection.suppression import SuppressionConfig, overlaps, suppress_overlaps

WHOLE = [0.0, 0.0, 100.0, 40.0]  # a whole fish
PART = [50.0, 5.0, 100.0, 35.0]  # its tail half, inside it: IoU 0.375, containment 1.0
NEIGHBOR = [90.0, 30.0, 190.0, 70.0]  # another fish, touching the first


def dets(boxes: list[list[float]], scores: list[float]) -> FrameDetections:
    return FrameDetections(np.asarray(boxes, dtype=np.float64), np.asarray(scores))


def test_overlaps_measures_iou_and_containment() -> None:
    iou, containment = overlaps(np.asarray([WHOLE, PART]))

    assert iou[0, 1] == pytest.approx(1500 / 4000)
    assert containment[0, 1] == pytest.approx(1.0)
    assert np.allclose(iou, iou.T) and np.allclose(containment, containment.T)


def test_zero_area_boxes_do_not_divide_by_zero() -> None:
    iou, containment = overlaps(np.asarray([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]]))

    assert np.all(np.isfinite(iou)) and np.all(np.isfinite(containment))


def test_disabled_suppression_returns_the_detections_unchanged() -> None:
    d = dets([WHOLE, PART], [0.9, 0.5])

    assert suppress_overlaps(d, SuppressionConfig()) is d


def test_containment_removes_a_part_box_that_iou_keeps() -> None:
    d = dets([PART, WHOLE, NEIGHBOR], [0.5, 0.9, 0.8])

    by_iou = suppress_overlaps(d, SuppressionConfig(iou=0.5))
    by_containment = suppress_overlaps(d, SuppressionConfig(containment=0.8))

    assert len(by_iou.scores) == 3
    # The lower-scoring part box goes; the neighbor stays; input order is kept.
    assert by_containment.scores.tolist() == [0.9, 0.8]
    assert by_containment.boxes.tolist() == [WHOLE, NEIGHBOR]


def test_a_suppressed_box_cannot_suppress_others() -> None:
    # A beats B, B would beat C, but C does not overlap A: greedy NMS keeps A and C.
    a, b, c = [0.0, 0, 10, 10], [6.0, 0, 16, 10], [12.0, 0, 22, 10]

    kept = suppress_overlaps(dets([a, b, c], [0.9, 0.8, 0.7]), SuppressionConfig(iou=0.2))

    assert kept.scores.tolist() == [0.9, 0.7]


def test_suppression_commutes_with_a_score_threshold() -> None:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 80, size=(40, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(5, 30, size=(40, 2))], axis=1)
    d = FrameDetections(boxes, rng.uniform(0.05, 1.0, size=40))
    config = SuppressionConfig(iou=0.4, containment=0.7)

    def above(x: FrameDetections, t: float) -> FrameDetections:
        return FrameDetections(x.boxes[x.scores >= t], x.scores[x.scores >= t])

    first = above(suppress_overlaps(d, config), 0.3)
    second = suppress_overlaps(above(d, 0.3), config)

    assert np.array_equal(first.boxes, second.boxes)


def test_thresholds_are_validated() -> None:
    with pytest.raises(ValidationError):
        SuppressionConfig(iou=1.5)
