"""ByteTrack: two-stage association that also uses low-score detections.

Implemented from the paper (Zhang et al., *ByteTrack: Multi-Object Tracking by Associating
Every Detection Box*, ECCV 2022); no ByteTrack code is vendored. Per frame:

1. every track predicts its box with a constant-velocity Kalman filter on
   ``(center x, center y, aspect ratio, height)``;
2. **high-score** detections (``score >= high_threshold``) are matched to all tracks, including
   recently lost ones, by IoU (``iou >= first_iou``);
3. **low-score** detections (``low_threshold <= score < high_threshold``) are matched to the
   tracks still unmatched that were active in the previous frame (``iou >= second_iou``).
   This recovers fish that briefly fade, instead of ending their tracks;
4. leftover high-score detections at or above ``new_track_threshold`` start new tracks;
5. a track unmatched for more than ``max_lost`` frames ends.

As in :mod:`passagewatch.tracking.kalman`, a trajectory contains only the detections actually
matched to it (never predicted boxes), and only tracks with at least ``min_length``
observations are returned. Track IDs are trajectories within one recording.

Differences from the reference implementation: no separate "unconfirmed track" matching
stage (``min_length`` filters short tracks at the end instead), plain IoU costs without
score fusion, and ``max_lost`` in frames rather than scaled from 30 fps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

from passagewatch.detection.classical import FrameDetections
from passagewatch.evaluation.detection import pairwise_iou
from passagewatch.tracking.kalman import Trajectory


class ByteTrackConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    high_threshold: float = Field(default=0.5, gt=0, le=1)
    low_threshold: float = Field(default=0.1, ge=0, le=1)
    new_track_threshold: float = Field(default=0.6, gt=0, le=1)
    first_iou: float = Field(default=0.2, ge=0, le=1)
    second_iou: float = Field(default=0.5, ge=0, le=1)
    max_lost: int = Field(default=8, ge=0)
    min_length: int = Field(default=3, ge=1)


# Constant-velocity model on (cx, cy, a, h, vcx, vcy, va, vh).
_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.eye(4, 8)
# Noise scaled by box height, as in SORT/ByteTrack.
_POS_WEIGHT = 1.0 / 20
_VEL_WEIGHT = 1.0 / 160


def _xyah(box: NDArray[np.float64]) -> NDArray[np.float64]:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, w / h, h])


def _box(state: NDArray[np.float64]) -> NDArray[np.float64]:
    cx, cy, a, h = state[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class _Track:
    track_id: int
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    lost: int = 0
    frames: list[int] = field(default_factory=list)
    boxes: list[NDArray[np.float64]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @classmethod
    def start(cls, track_id: int, box: NDArray[np.float64], score: float, frame: int) -> _Track:
        measurement = _xyah(box)
        h = measurement[3]
        std = np.array(
            [
                2 * _POS_WEIGHT * h,
                2 * _POS_WEIGHT * h,
                1e-2,
                2 * _POS_WEIGHT * h,
                10 * _VEL_WEIGHT * h,
                10 * _VEL_WEIGHT * h,
                1e-5,
                10 * _VEL_WEIGHT * h,
            ]
        )
        return cls(
            track_id=track_id,
            mean=np.concatenate([measurement, np.zeros(4)]),
            covariance=np.diag(std**2),
            frames=[frame],
            boxes=[box],
            scores=[score],
        )

    def predict(self) -> None:
        h = self.mean[3]
        std = np.array(
            [
                _POS_WEIGHT * h,
                _POS_WEIGHT * h,
                1e-2,
                _POS_WEIGHT * h,
                _VEL_WEIGHT * h,
                _VEL_WEIGHT * h,
                1e-5,
                _VEL_WEIGHT * h,
            ]
        )
        self.mean = _F @ self.mean
        self.covariance = _F @ self.covariance @ _F.T + np.diag(std**2)

    def update(self, box: NDArray[np.float64], score: float, frame: int) -> None:
        h = self.mean[3]
        std = np.array([_POS_WEIGHT * h, _POS_WEIGHT * h, 1e-1, _POS_WEIGHT * h])
        s = _H @ self.covariance @ _H.T + np.diag(std**2)
        gain = self.covariance @ _H.T @ np.linalg.inv(s)
        self.mean = self.mean + gain @ (_xyah(box) - _H @ self.mean)
        self.covariance = (np.eye(8) - gain @ _H) @ self.covariance
        self.lost = 0
        self.frames.append(frame)
        self.boxes.append(box)
        self.scores.append(score)

    @property
    def predicted_box(self) -> NDArray[np.float64]:
        return _box(self.mean)


def _match(
    tracks: list[_Track], boxes: NDArray[np.float64], min_iou: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching on IoU; returns matches, unmatched tracks, unmatched detections."""
    if not tracks or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    predicted = np.stack([t.predicted_box for t in tracks])
    iou = pairwise_iou(predicted, boxes)
    cost = np.where(iou >= min_iou, 1.0 - iou, 1e6)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r, c] < 1e6]
    matched_t = {r for r, _ in matches}
    matched_d = {c for _, c in matches}
    return (
        matches,
        [i for i in range(len(tracks)) if i not in matched_t],
        [j for j in range(len(boxes)) if j not in matched_d],
    )


class ByteTracker:
    def __init__(self, config: ByteTrackConfig) -> None:
        if config.low_threshold > config.high_threshold:
            raise ValueError("low_threshold must not exceed high_threshold")
        self.config = config

    def run(self, detections: list[FrameDetections], frame_offset: int = 0) -> list[Trajectory]:
        """Track a whole recording; ``detections[t]`` belong to frame ``frame_offset + t``."""
        cfg = self.config
        tracks: list[_Track] = []  # active and recently lost
        finished: list[_Track] = []
        next_id = 1
        for t, frame in enumerate(detections):
            frame_index = frame_offset + t
            for track in tracks:
                track.predict()
            high = frame.scores >= cfg.high_threshold
            low = (frame.scores >= cfg.low_threshold) & ~high
            high_boxes, high_scores = frame.boxes[high], frame.scores[high]
            low_boxes, low_scores = frame.boxes[low], frame.scores[low]

            # 1. High-score detections against every track.
            matches, unmatched_t, unmatched_d = _match(tracks, high_boxes, cfg.first_iou)
            for ti, di in matches:
                tracks[ti].update(high_boxes[di], float(high_scores[di]), frame_index)

            # 2. Low-score detections against tracks active in the previous frame.
            remaining = [tracks[i] for i in unmatched_t if tracks[i].lost == 0]
            second, _, _ = _match(remaining, low_boxes, cfg.second_iou)
            for ti, di in second:
                remaining[ti].update(low_boxes[di], float(low_scores[di]), frame_index)
            updated = {id(tracks[ti]) for ti, _ in matches} | {
                id(remaining[ti]) for ti, _ in second
            }

            # 3. Unmatched tracks age; those lost too long end.
            kept: list[_Track] = []
            for track in tracks:
                if id(track) not in updated:
                    track.lost += 1
                (finished if track.lost > cfg.max_lost else kept).append(track)
            tracks = kept

            # 4. Confident leftover detections start new tracks.
            for di in unmatched_d:
                if high_scores[di] >= cfg.new_track_threshold:
                    tracks.append(
                        _Track.start(next_id, high_boxes[di], float(high_scores[di]), frame_index)
                    )
                    next_id += 1
        finished.extend(tracks)

        trajectories = [
            Trajectory(
                track_id=track.track_id,
                frames=np.asarray(track.frames, dtype=np.int64),
                boxes=np.asarray(track.boxes, dtype=np.float64).reshape(-1, 4),
                scores=np.asarray(track.scores, dtype=np.float64),
            )
            for track in finished
            if len(track.frames) >= cfg.min_length
        ]
        trajectories.sort(key=lambda tr: tr.track_id)
        return trajectories
