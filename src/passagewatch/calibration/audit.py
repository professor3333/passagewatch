"""Random audit windows over footage that the review queue does not cover.

A queue built from predicted trajectories cannot show a fish the detector missed entirely.
So each recording also gets a few randomly placed windows of **unflagged** footage (no
counted or flagged trajectory nearby) for a reviewer to watch. They make it possible to
estimate the rate of fish missed outright, which reviewing the queue never reveals.

Windows are drawn reproducibly from a seed (a job's ID), never overlap each other or the
flagged intervals (widened by ``margin_frames``), and are added until they cover at least
``fraction`` of the unflagged frames. A recording with less unflagged footage than one
window gets one window covering its longest unflagged stretch, if any.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

AUDIT_VERSION = "audit-v0"


@dataclass(frozen=True)
class AuditWindow:
    start_frame: int  # inclusive
    stop_frame: int  # exclusive

    @property
    def frames(self) -> int:
        return self.stop_frame - self.start_frame


def seed_from(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def unflagged_mask(
    num_frames: int, flagged: Sequence[tuple[int, int]], margin_frames: int
) -> np.ndarray:
    """``True`` for frames outside every flagged ``[first, last]`` interval (inclusive),
    each widened by ``margin_frames`` on both sides."""
    mask = np.ones(num_frames, dtype=bool)
    for first, last in flagged:
        lo = max(0, first - margin_frames)
        hi = max(0, min(num_frames, last + 1 + margin_frames))
        mask[lo:hi] = False
    return mask


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` runs of True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def audit_windows(
    num_frames: int,
    flagged: Sequence[tuple[int, int]],
    *,
    window_frames: int,
    fraction: float,
    seed: int,
    margin_frames: int = 0,
) -> list[AuditWindow]:
    if num_frames < 0 or window_frames < 1 or not 0 < fraction <= 1:
        raise ValueError("invalid audit parameters")
    mask = unflagged_mask(num_frames, flagged, margin_frames)
    total = int(mask.sum())
    if total == 0:
        return []
    runs = _runs(mask)
    if all(stop - start < window_frames for start, stop in runs):
        start, stop = max(runs, key=lambda r: (r[1] - r[0], -r[0]))
        return [AuditWindow(start, stop)]
    rng = np.random.default_rng(seed)
    free = mask.copy()
    chosen: list[AuditWindow] = []
    covered = 0
    while covered < fraction * total:
        # Every start whose whole window is still free, each equally likely.
        starts = [s for start, stop in _runs(free) for s in range(start, stop - window_frames + 1)]
        if not starts:
            break
        s = int(starts[int(rng.integers(len(starts)))])
        chosen.append(AuditWindow(s, s + window_frames))
        free[s : s + window_frames] = False
        covered += window_frames
    return sorted(chosen, key=lambda w: w.start_frame)
