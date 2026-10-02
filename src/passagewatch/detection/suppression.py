"""Extra duplicate suppression after the detector's own NMS.

YOLOX's NMS (IoU 0.65) keeps a box covering part of a fish next to the box covering the
whole fish: their IoU is often 0.4-0.65, but most of the smaller box lies inside the larger
one (docs/error_analysis.md). The tracker then follows both, and the fish is counted twice.

:func:`suppress_overlaps` is a second greedy pass in descending score order. A box is
dropped when a kept, higher-scoring box overlaps it with IoU above ``iou`` or with
*containment* (intersection over the smaller box's area) above ``containment``. Either
criterion can be off. Greedy suppression in score order commutes with a score threshold,
so applying it to cached detections and then thresholding gives the same boxes as
thresholding first.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from passagewatch.detection.classical import FrameDetections


class SuppressionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    iou: float | None = Field(default=None, gt=0, lt=1)
    containment: float | None = Field(default=None, gt=0, lt=1)

    @property
    def enabled(self) -> bool:
        return self.iou is not None or self.containment is not None


def overlaps(boxes: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pairwise IoU and containment (intersection over the smaller area) of ``(n, 4)`` boxes."""
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    union = area[:, None] + area[None, :] - inter
    smaller = np.minimum(area[:, None], area[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
        containment = np.where(smaller > 0, inter / smaller, 0.0)
    return iou, containment


def suppress_overlaps(detections: FrameDetections, config: SuppressionConfig) -> FrameDetections:
    """Drop boxes that duplicate a higher-scoring box; the kept boxes stay in input order."""
    n = len(detections.scores)
    if not config.enabled or n < 2:
        return detections
    order = np.argsort(-detections.scores, kind="stable")
    iou, containment = overlaps(detections.boxes[order])
    duplicate = np.zeros((n, n), dtype=bool)
    if config.iou is not None:
        duplicate |= iou > config.iou
    if config.containment is not None:
        duplicate |= containment > config.containment
    suppressed = np.zeros(n, dtype=bool)
    kept = []
    for i in range(n):
        if suppressed[i]:
            continue
        kept.append(i)
        suppressed[i + 1 :] |= duplicate[i, i + 1 :]
    keep = np.sort(order[kept])
    return FrameDetections(detections.boxes[keep], detections.scores[keep])
