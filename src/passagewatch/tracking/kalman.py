"""A simple multi-object tracker: constant-velocity Kalman filters and gated Hungarian matching.

Each track's state is its center position and velocity in meters (``x, y, vx, vy``), so the
association gate means the same distance on every camera. Every frame:

1. all tracks predict one frame ahead;
2. detections are assigned to tracks by minimum total distance (Hungarian), but only pairs
   closer than ``gate_m`` are allowed;
3. matched tracks are corrected; unmatched detections start new tentative tracks;
4. a track that has missed more than ``max_age`` consecutive frames ends.

A track is **confirmed** after ``min_hits`` matched detections. When the recording ends, all
tracks end. Confirmed tracks with at least ``min_length`` observations are returned. A
trajectory's observations are only the detections actually matched to it; predicted
positions are never reported as boxes. Track IDs are trajectories within one recording,
not fish identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

from passagewatch.detection.classical import FrameDetections
from passagewatch.ingestion.mot import BoxAnnotations


class TrackerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    gate_m: float = Field(default=0.5, gt=0)
    max_age: int = Field(default=3, ge=0)
    min_hits: int = Field(default=3, ge=1)
    min_length: int = Field(default=3, ge=1)
    process_noise: float = Field(default=0.05, gt=0)
    measurement_noise: float = Field(default=0.05, gt=0)


_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)


@dataclass
class _Track:
    track_id: int
    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    frames: list[int] = field(default_factory=list)
    boxes: list[NDArray[np.float64]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    misses: int = 0

    @property
    def position(self) -> NDArray[np.float64]:
        return self.state[:2]


@dataclass(frozen=True)
class Trajectory:
    track_id: int
    frames: NDArray[np.int64]
    boxes: NDArray[np.float64]
    scores: NDArray[np.float64]


class KalmanTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        q = config.process_noise**2
        self._Q = np.diag([q, q, q, q])
        r = config.measurement_noise**2
        self._R = np.diag([r, r])

    def run(
        self,
        detections: list[FrameDetections],
        centers_m: list[NDArray[np.float64]],
        frame_offset: int = 0,
    ) -> list[Trajectory]:
        """Track a whole recording. ``centers_m[t]`` are detection centers in meters."""
        active: list[_Track] = []
        finished: list[_Track] = []
        next_id = 1
        for t, (frame, centers) in enumerate(zip(detections, centers_m, strict=True)):
            for track in active:
                track.state = _F @ track.state
                track.covariance = _F @ track.covariance @ _F.T + self._Q

            matches, unmatched = self._associate(active, centers)
            for ti, di in matches:
                track = active[ti]
                innovation = centers[di] - _H @ track.state
                s = _H @ track.covariance @ _H.T + self._R
                gain = track.covariance @ _H.T @ np.linalg.inv(s)
                track.state = track.state + gain @ innovation
                track.covariance = (np.eye(4) - gain @ _H) @ track.covariance
                track.frames.append(frame_offset + t)
                track.boxes.append(frame.boxes[di])
                track.scores.append(float(frame.scores[di]))
                track.misses = 0
            matched_tracks = {ti for ti, _ in matches}
            still_active: list[_Track] = []
            for ti, track in enumerate(active):
                if ti not in matched_tracks:
                    track.misses += 1
                (finished if track.misses > self.config.max_age else still_active).append(track)
            active = still_active

            for di in unmatched:
                active.append(
                    _Track(
                        track_id=next_id,
                        state=np.array([*centers[di], 0.0, 0.0]),
                        covariance=np.diag([0.01, 0.01, 1.0, 1.0]),
                        frames=[frame_offset + t],
                        boxes=[frame.boxes[di]],
                        scores=[float(frame.scores[di])],
                    )
                )
                next_id += 1
        finished.extend(active)

        keep = max(self.config.min_hits, self.config.min_length)
        trajectories = [
            Trajectory(
                track_id=track.track_id,
                frames=np.asarray(track.frames, dtype=np.int64),
                boxes=np.asarray(track.boxes, dtype=np.float64).reshape(-1, 4),
                scores=np.asarray(track.scores, dtype=np.float64),
            )
            for track in finished
            if len(track.frames) >= keep
        ]
        trajectories.sort(key=lambda tr: tr.track_id)
        return trajectories

    def _associate(
        self, tracks: list[_Track], centers: NDArray[np.float64]
    ) -> tuple[list[tuple[int, int]], list[int]]:
        if not tracks or len(centers) == 0:
            return [], list(range(len(centers)))
        predicted = np.stack([t.position for t in tracks])
        cost = np.linalg.norm(predicted[:, None, :] - centers[None, :, :], axis=2)
        gated = np.where(cost <= self.config.gate_m, cost, 1e6)
        rows, cols = linear_sum_assignment(gated)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if gated[r, c] < 1e6]
        matched = {c for _, c in matches}
        return matches, [i for i in range(len(centers)) if i not in matched]


def trajectories_to_annotations(trajectories: list[Trajectory]) -> BoxAnnotations:
    """Flatten trajectories into internal-convention box annotations (renumbered from 1)."""
    if not trajectories:
        return BoxAnnotations.empty()
    return BoxAnnotations(
        frame_index=np.concatenate([tr.frames for tr in trajectories]),
        track_id=np.concatenate(
            [np.full(len(tr.frames), i, dtype=np.int64) for i, tr in enumerate(trajectories, 1)]
        ),
        boxes=np.concatenate([tr.boxes for tr in trajectories]),
    )
