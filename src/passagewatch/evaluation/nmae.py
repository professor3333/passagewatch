"""Directional normalized mean absolute error (nMAE), per location (``docs/counting_policy.md`` §5).

``nMAE(group) = Σ_clips (|R̂ - R| + |L̂ - L|) / Σ_clips (R + L)``

Directions are kept separate and never netted. Reference counts come from applying the
same counting policy to the reference trajectories. When a group has no true passages,
nMAE is undefined (the official code divides by zero); the absolute error and the
predicted passages per hour are reported instead.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from passagewatch.counting.policy import (
    CFC_COMPATIBLE_V1,
    CountingPolicy,
    DirectionalCounts,
    count_trajectories,
    tally,
)
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.mot import BoxAnnotations, read_mot


@dataclass(frozen=True)
class ClipCountError:
    clip_name: str
    reference: DirectionalCounts
    predicted: DirectionalCounts
    duration_seconds: float

    @property
    def absolute_error(self) -> int:
        """The nMAE numerator for this clip: per-direction errors, summed."""
        return abs(self.predicted.right - self.reference.right) + abs(
            self.predicted.left - self.reference.left
        )


@dataclass(frozen=True)
class GroupResult:
    group: str
    clips: int
    absolute_error: int
    reference_passages: int
    predicted_passages: int
    hours: float

    @property
    def nmae(self) -> float | None:
        if self.reference_passages == 0:
            return None
        return self.absolute_error / self.reference_passages

    @property
    def predicted_passages_per_hour(self) -> float:
        return self.predicted_passages / self.hours if self.hours > 0 else math.nan


def summarize(group: str, errors: Iterable[ClipCountError]) -> GroupResult:
    errors = list(errors)
    return GroupResult(
        group=group,
        clips=len(errors),
        absolute_error=sum(e.absolute_error for e in errors),
        reference_passages=sum(e.reference.total for e in errors),
        predicted_passages=sum(e.predicted.total for e in errors),
        hours=sum(e.duration_seconds for e in errors) / 3600.0,
    )


def macro_nmae(groups: Iterable[GroupResult]) -> float | None:
    """Mean of the defined per-group nMAE values; None if no group defines one."""
    values = [g.nmae for g in groups if g.nmae is not None]
    return sum(values) / len(values) if values else None


def count_clip(
    annotations: BoxAnnotations, meta: ClipMetadata, policy: CountingPolicy
) -> DirectionalCounts:
    outside = (annotations.frame_index < 0) | (annotations.frame_index >= meta.num_frames)
    if outside.any():
        raise ValueError(
            f"{meta.clip_name}: {int(outside.sum())} boxes outside frames "
            f"[1, {meta.num_frames}] (1-based)"
        )
    return tally(count_trajectories(annotations, meta.width, meta.height, policy))


def evaluate_mot_location(
    reference_dir: Path,
    predicted_dir: Path,
    metadata: dict[str, ClipMetadata],
    policy: CountingPolicy = CFC_COMPATIBLE_V1,
) -> list[ClipCountError]:
    """Compare MOT-format predictions with reference trajectories for every clip.

    ``reference_dir/<clip>/gt.txt`` holds the reference and ``predicted_dir/<clip>.txt`` the
    predictions (the MOTChallenge layout used by CFC). Every clip in ``metadata`` must have
    a prediction file; a missing one is an error rather than a silent zero.
    """
    errors: list[ClipCountError] = []
    for name in sorted(metadata):
        meta = metadata[name]
        predicted_path = predicted_dir / f"{name}.txt"
        if not predicted_path.is_file():
            raise FileNotFoundError(f"no prediction file for clip {name}: {predicted_path}")
        errors.append(
            ClipCountError(
                clip_name=name,
                reference=count_clip(read_mot(reference_dir / name / "gt.txt"), meta, policy),
                predicted=count_clip(read_mot(predicted_path), meta, policy),
                duration_seconds=meta.num_frames / meta.framerate,
            )
        )
    return errors
