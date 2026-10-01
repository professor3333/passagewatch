"""The ``cfc-compatible-v1`` counting policy (``docs/counting_policy.md`` §2).

Counts come only from **completed** trajectories, never from per-frame detections. Each
trajectory contributes at most one passage, decided by its first and last box centers:

- the start-to-end distance in per-axis normalized coordinates must be at least
  ``min_displacement_normalized`` (stationary filter; anisotropic, as in the official code);
- ``u0 < line`` and ``u1 >= line`` is one rightward passage; ``u0 >= line`` and
  ``u1 < line`` is one leftward passage; anything else contributes nothing.

So a fish that crosses and returns to its starting side contributes zero.

The arithmetic follows the official evaluator's order of operations (normalize the box,
then take its center), so that borderline comparisons give identical results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from passagewatch.ingestion.mot import BoxAnnotations


class Direction(StrEnum):
    """Image-space direction of travel; the primary output."""

    RIGHT = "right"
    LEFT = "left"


class Outcome(StrEnum):
    PASSAGE = "passage"
    STATIONARY = "stationary"
    NO_CROSSING = "no_crossing"


@dataclass(frozen=True)
class CountingPolicy:
    version: str
    line_x_normalized: float = 0.5
    min_displacement_normalized: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.line_x_normalized < 1.0:
            raise ValueError(f"line_x_normalized must be in (0, 1): {self.line_x_normalized}")
        if self.min_displacement_normalized < 0.0:
            raise ValueError("min_displacement_normalized must be >= 0")


CFC_COMPATIBLE_V1 = CountingPolicy(version="cfc-compatible-v1")


def is_benchmark_compatible(policy: CountingPolicy) -> bool:
    """True only for the exact parameters of the official CFC evaluator."""
    return policy == CFC_COMPATIBLE_V1


@dataclass(frozen=True)
class TrajectoryCount:
    """How one completed trajectory was counted, kept as evidence for review."""

    track_id: int
    start_frame: int
    end_frame: int
    observations: int
    start_u: float
    end_u: float
    displacement: float
    outcome: Outcome
    direction: Direction | None


@dataclass(frozen=True)
class DirectionalCounts:
    right: int = 0
    left: int = 0

    @property
    def total(self) -> int:
        return self.right + self.left

    def __add__(self, other: DirectionalCounts) -> DirectionalCounts:
        return DirectionalCounts(self.right + other.right, self.left + other.left)


def _normalized_center(box: np.ndarray, width: int, height: int) -> tuple[float, float]:
    # Official order: normalize origin and size first, then add half the size.
    x_min, y_min, x_max, y_max = (float(v) for v in box)
    u = x_min / width + ((x_max - x_min) / width) / 2.0
    v = y_min / height + ((y_max - y_min) / height) / 2.0
    return u, v


def count_trajectories(
    annotations: BoxAnnotations,
    width: int,
    height: int,
    policy: CountingPolicy = CFC_COMPATIBLE_V1,
) -> list[TrajectoryCount]:
    """Apply ``policy`` to every trajectory in ``annotations``, ordered by track ID.

    ``width`` and ``height`` are the frame size used for normalization (the clip metadata
    size for CFC). A trajectory with two boxes in one frame is an error.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    results: list[TrajectoryCount] = []
    for track in np.unique(annotations.track_id).tolist():
        rows = np.flatnonzero(annotations.track_id == track)
        frames = annotations.frame_index[rows]
        if len(np.unique(frames)) != len(frames):
            raise ValueError(f"track {track} has more than one box in a frame")
        order = rows[np.argsort(frames, kind="stable")]
        first, last = order[0], order[-1]
        u0, v0 = _normalized_center(annotations.boxes[first], width, height)
        u1, v1 = _normalized_center(annotations.boxes[last], width, height)
        displacement = math.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2)

        line = policy.line_x_normalized
        direction: Direction | None = None
        if policy.min_displacement_normalized > 0 and (
            displacement < policy.min_displacement_normalized
        ):
            outcome = Outcome.STATIONARY
        elif u0 < line <= u1:
            outcome, direction = Outcome.PASSAGE, Direction.RIGHT
        elif u1 < line <= u0:
            outcome, direction = Outcome.PASSAGE, Direction.LEFT
        else:
            outcome = Outcome.NO_CROSSING
        results.append(
            TrajectoryCount(
                track_id=int(track),
                start_frame=int(annotations.frame_index[first]),
                end_frame=int(annotations.frame_index[last]),
                observations=len(rows),
                start_u=u0,
                end_u=u1,
                displacement=displacement,
                outcome=outcome,
                direction=direction,
            )
        )
    return results


def tally(trajectories: list[TrajectoryCount]) -> DirectionalCounts:
    right = sum(t.direction is Direction.RIGHT for t in trajectories)
    left = sum(t.direction is Direction.LEFT for t in trajectories)
    return DirectionalCounts(right=right, left=left)


@dataclass(frozen=True)
class RiverCounts:
    upstream: int
    downstream: int

    @property
    def net(self) -> int:
        return self.upstream - self.downstream


def to_river_directions(
    counts: DirectionalCounts, upstream_direction: Direction | None
) -> RiverCounts | None:
    """Map image directions to upstream/downstream, only when orientation is configured."""
    if upstream_direction is None:
        return None
    if upstream_direction is Direction.RIGHT:
        return RiverCounts(upstream=counts.right, downstream=counts.left)
    return RiverCounts(upstream=counts.left, downstream=counts.right)
