"""Calibration versions: how a release scores trajectories for review and samples audits.

A release bundle names one in ``calibration_version`` (or none). ``review-v0`` is
``review-score-v0`` with its default thresholds plus ``audit-v0`` random audits covering at
least 10% of the unflagged frames in 5 s windows, kept 1 s away from flagged trajectories.
It is heuristic: no probability is fitted. A changed setting needs a new version.

:func:`review_clip` is the single implementation, used by the worker and by
``scripts/evaluate_review.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from passagewatch.calibration.audit import AuditWindow, audit_windows, unflagged_mask
from passagewatch.calibration.review_score import (
    ReviewConfig,
    TrackReview,
    review,
    track_features,
)
from passagewatch.counting.policy import TrajectoryCount
from passagewatch.tracking.kalman import Trajectory

REVIEW_V0 = "review-v0"


@dataclass(frozen=True)
class Calibration:
    version: str
    review: ReviewConfig
    audit_fraction: float
    audit_window_s: float
    audit_margin_s: float


CALIBRATIONS = {
    REVIEW_V0: Calibration(
        version=REVIEW_V0,
        review=ReviewConfig(),
        audit_fraction=0.1,
        audit_window_s=5.0,
        audit_margin_s=1.0,
    ),
}


def get_calibration(version: str | None) -> Calibration | None:
    if version is None:
        return None
    try:
        return CALIBRATIONS[version]
    except KeyError:
        raise ValueError(f"unknown calibration version {version!r}") from None


@dataclass(frozen=True)
class ClipReview:
    tracks: dict[int, TrackReview]
    audit: list[AuditWindow]
    unflagged_frames: int


def is_flagged(r: TrackReview) -> bool:
    """A trajectory a reviewer already sees: every passage, and anything not suggested."""
    return r.features.is_passage or r.triage != "suggested"


def review_clip(
    calibration: Calibration,
    trajectories: Sequence[Trajectory],
    counts: Sequence[TrajectoryCount],
    *,
    meters_per_px: tuple[float, float],
    line_x_normalized: float,
    num_frames: int,
    framerate: float,
    frame_offset: int = 0,
    seed: int,
) -> ClipReview:
    """Review scores and triage for every trajectory, and the clip's audit windows.

    Trajectory frames are ``frame_offset`` + a 0-based index into the clip; audit windows
    are 0-based.
    """
    features = track_features(trajectories, counts, meters_per_px, line_x_normalized)
    reviews = {f.track_id: review(f, calibration.review) for f in features}
    flagged = [
        (int(t.frames.min()) - frame_offset, int(t.frames.max()) - frame_offset)
        for t in trajectories
        if is_flagged(reviews[t.track_id])
    ]
    margin = round(calibration.audit_margin_s * framerate)
    windows = audit_windows(
        num_frames,
        flagged,
        window_frames=max(1, round(calibration.audit_window_s * framerate)),
        fraction=calibration.audit_fraction,
        seed=seed,
        margin_frames=margin,
    )
    unflagged = int(unflagged_mask(num_frames, flagged, margin).sum())
    return ClipReview(reviews, windows, unflagged)
