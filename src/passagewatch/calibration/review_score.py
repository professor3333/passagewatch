"""A heuristic review score for each trajectory, and its triage state.

The detector's score is local evidence about one box; it is not the probability that a count
is correct. A trajectory's review score combines the evidence the project plan lists, each
mapped to ``[0, 1]`` (1 = strong) with fixed, declared anchors:

| Component | Feature | 0 at | 1 at |
|---|---|---|---|
| ``detection`` | mean detector score | 0.2 | 0.6 |
| ``duration`` | observations | 3 | 15 |
| ``continuity`` | gap fraction, ``1 - observations / span`` | 0.5 | 0 |
| ``motion`` | straightness: net displacement / path length (m) | 0 | 1 |
| ``separation`` | share of frames with no other box at IoU > 0.1 | 0 | 1 |
| ``line_distance`` | normalized x distance of the nearer endpoint from the line | 0 | 0.15 |

``review_score`` is their geometric mean, with each component floored at ``0.01``. Unlike an
arithmetic mean, one weak component (a short track, or weak detections) lowers it sharply
instead of being averaged away by the others. It is a **ranking heuristic, not a
probability**, and the UI must call it a review score until a calibrator has been validated
(``CLAUDE.md``).

Triage (``REVIEW_SCORE_VERSION``; thresholds are provisional until measured in Stage 9):

- ``unresolved``: a passage whose direction is not established by its evidence: its path
  is mostly back and forth (straightness below ``unresolved_straightness``), or it starts or
  ends within ``unresolved_line_margin`` of the line, so whether it crossed depends on a few
  pixels.
- ``needs_review``: a passage with ``review_score`` below ``suggest_threshold``, or a
  non-passing track that starts or ends near the line (``candidate_line_margin``) and lasts
  at least ``candidate_min_observations``: it may be a fragment of a missed passage.
- ``suggested``: everything else.

Triage never changes the automatic counts: those come only from the counting policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from passagewatch.counting.policy import TrajectoryCount
from passagewatch.tracking.kalman import Trajectory

REVIEW_SCORE_VERSION = "review-score-v0"
Triage = Literal["suggested", "needs_review", "unresolved"]
COMPONENT_FLOOR = 0.01
COMPONENTS = ("detection", "duration", "continuity", "motion", "separation", "line_distance")


class ReviewConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    suggest_threshold: float = Field(default=0.6, ge=0, le=1)
    unresolved_straightness: float = Field(default=0.3, ge=0, le=1)
    unresolved_line_margin: float = Field(default=0.02, ge=0, lt=0.5)
    candidate_line_margin: float = Field(default=0.05, ge=0, lt=0.5)
    candidate_min_observations: int = Field(default=5, ge=1)


@dataclass(frozen=True)
class TrackFeatures:
    track_id: int
    mean_score: float
    observations: int
    gap_fraction: float
    straightness: float
    crowding: float  # fraction of observed frames with another track's box overlapping
    line_margin: float  # normalized x distance of the nearer endpoint from the line
    span_frames: int
    is_passage: bool


@dataclass(frozen=True)
class TrackReview:
    features: TrackFeatures
    components: dict[str, float]
    review_score: float
    triage: Triage
    reasons: tuple[str, ...]


def _ramp(value: float, zero: float, one: float) -> float:
    """``value`` mapped linearly so that ``zero`` -> 0 and ``one`` -> 1, clipped to [0, 1]."""
    return float(np.clip((value - zero) / (one - zero), 0.0, 1.0))


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def track_features(
    trajectories: Sequence[Trajectory],
    counts: Sequence[TrajectoryCount],
    meters_per_px: tuple[float, float],
    line_x_normalized: float,
) -> list[TrackFeatures]:
    """Features of every trajectory; ``counts`` are the counting policy's decisions for the
    same trajectories (matched by track ID)."""
    by_id = {c.track_id: c for c in counts}
    boxes_by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for t in trajectories:
        for frame, box in zip(t.frames.tolist(), t.boxes, strict=True):
            boxes_by_frame.setdefault(frame, []).append((t.track_id, box))
    sx, sy = meters_per_px
    out = []
    for t in trajectories:
        count = by_id[t.track_id]
        n = len(t.frames)
        span = int(t.frames.max() - t.frames.min() + 1) if n else 0
        centers = np.stack(
            [(t.boxes[:, 0] + t.boxes[:, 2]) / 2 * sx, (t.boxes[:, 1] + t.boxes[:, 3]) / 2 * sy],
            axis=1,
        )
        path = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()) if n > 1 else 0.0
        net = float(np.linalg.norm(centers[-1] - centers[0])) if n > 1 else 0.0
        crowded = 0
        for frame, box in zip(t.frames.tolist(), t.boxes, strict=True):
            others = [b for tid, b in boxes_by_frame[frame] if tid != t.track_id]
            if others and float(_iou(box, np.stack(others)).max()) > 0.1:
                crowded += 1
        out.append(
            TrackFeatures(
                track_id=t.track_id,
                mean_score=float(t.scores.mean()) if n else 0.0,
                observations=n,
                gap_fraction=1.0 - n / span if span else 0.0,
                straightness=net / path if path > 0 else 1.0,
                crowding=crowded / n if n else 0.0,
                line_margin=min(
                    abs(count.start_u - line_x_normalized), abs(count.end_u - line_x_normalized)
                ),
                span_frames=span,
                is_passage=count.direction is not None,
            )
        )
    return out


def review(features: TrackFeatures, config: ReviewConfig | None = None) -> TrackReview:
    config = config or ReviewConfig()
    f = features
    components = {
        "detection": _ramp(f.mean_score, 0.2, 0.6),
        "duration": _ramp(f.observations, 3, 15),
        "continuity": 1.0 - _ramp(f.gap_fraction, 0.0, 0.5),
        "motion": float(np.clip(f.straightness, 0.0, 1.0)),
        "separation": 1.0 - float(np.clip(f.crowding, 0.0, 1.0)),
        "line_distance": _ramp(f.line_margin, 0.0, 0.15),
    }
    floored = np.maximum([components[c] for c in COMPONENTS], COMPONENT_FLOOR)
    score = float(np.exp(np.log(floored).mean()))
    reasons = tuple(c for c in COMPONENTS if components[c] < 0.5)
    triage: Triage
    if f.is_passage:
        if f.straightness < config.unresolved_straightness:
            triage, reasons = "unresolved", ("back_and_forth", *reasons)
        elif f.line_margin < config.unresolved_line_margin:
            triage, reasons = "unresolved", ("endpoint_on_line", *reasons)
        elif score < config.suggest_threshold:
            triage = "needs_review"
        else:
            triage = "suggested"
    elif (
        f.line_margin < config.candidate_line_margin
        and f.observations >= config.candidate_min_observations
    ):
        triage, reasons = "needs_review", ("possible_missed_passage", *reasons)
    else:
        triage = "suggested"
    return TrackReview(f, components, score, triage, reasons)
