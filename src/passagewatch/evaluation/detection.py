"""Detection-level diagnostics: how many reference boxes the detector found, per frame.

This is a tuning aid, not the release metric (that is counting nMAE). Within each frame,
detections are matched one to one with reference boxes, greedily by descending IoU, and a
pair counts only at ``iou >= iou_threshold``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from passagewatch.detection.classical import FrameDetections
from passagewatch.ingestion.mot import BoxAnnotations


@dataclass(frozen=True)
class DetectionMatch:
    matched: int
    reference_boxes: int
    detections: int

    @property
    def recall(self) -> float | None:
        return self.matched / self.reference_boxes if self.reference_boxes else None

    @property
    def precision(self) -> float | None:
        return self.matched / self.detections if self.detections else None

    def __add__(self, other: DetectionMatch) -> DetectionMatch:
        return DetectionMatch(
            self.matched + other.matched,
            self.reference_boxes + other.reference_boxes,
            self.detections + other.detections,
        )


def pairwise_iou(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """IoU matrix ``(len(a), len(b))`` for internal-convention boxes."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match_frame(
    reference: NDArray[np.float64], predicted: NDArray[np.float64], iou_threshold: float
) -> int:
    if len(reference) == 0 or len(predicted) == 0:
        return 0
    iou = pairwise_iou(reference, predicted)
    matched = 0
    used_r: set[int] = set()
    used_p: set[int] = set()
    for flat in np.argsort(-iou, axis=None):
        r, p = divmod(int(flat), iou.shape[1])
        if iou[r, p] < iou_threshold:
            break
        if r in used_r or p in used_p:
            continue
        used_r.add(r)
        used_p.add(p)
        matched += 1
    return matched


def match_detections(
    reference: BoxAnnotations,
    detections: Sequence[FrameDetections],
    frame_start: int,
    iou_threshold: float = 0.3,
) -> DetectionMatch:
    """Match ``detections[t]`` (frame ``frame_start + t``) against the reference boxes."""
    total = DetectionMatch(0, 0, 0)
    for t, frame in enumerate(detections):
        ref = reference.boxes[reference.frame_index == frame_start + t]
        total = total + DetectionMatch(
            matched=match_frame(ref, frame.boxes, iou_threshold),
            reference_boxes=len(ref),
            detections=len(frame.boxes),
        )
    return total
