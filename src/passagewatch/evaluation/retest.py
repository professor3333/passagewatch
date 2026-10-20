"""The same-fish question re-test (docs/review_retest.md).

New reviewers review 21 kenai-holdout-v1 clips with the review page's same-fish question:
set A, the reviewed sample's ``s01``-``s15`` (reviewed by R1 and R2 without the question),
and set B, the first six clips of ``s16``-``s50`` whose automatic counts are below the
reference. The clips are shown as ``t01``-``t21`` in one order drawn with seed 0.

Two targets, judged per reviewer, each met when it holds for at least two thirds of the
reviewers who took part (at least 3):

1. **No harm:** the reviewed counts' error against the reference is at most the automatic
   counts' error, over the reviewer's complete clips.
2. **Fewer fragment counts:** at most ``MAX_FRAGMENT_COUNTS`` on the complete clips of set A.
   A fragment count counts a track the release did not count, whose first and last boxes lie
   on the same side of the line, next to a track the release counted (``describe_changes``).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from passagewatch.evaluation.same_fish import SAME_FISH_REASONS

SEED = 0
SET_A = tuple(f"s{i:02d}" for i in range(1, 16))
SET_B_SIZE = 6
REVIEWERS = ("T1", "T2", "T3", "T4")
MAX_FRAGMENT_COUNTS = 3
MIN_REVIEWERS = 3


def set_b(sample_clips: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """The first ``SET_B_SIZE`` codes of ``s16``-``s50``, in code order, whose automatic
    counts are below the reference in either direction."""
    chosen = [
        code
        for code in sorted(sample_clips)
        if code >= "s16"
        and any(
            a < r
            for a, r in zip(
                sample_clips[code]["automatic"], sample_clips[code]["reference"], strict=True
            )
        )
    ]
    if len(chosen) < SET_B_SIZE:
        raise ValueError(f"only {len(chosen)} undercounted clips in s16-s50")
    return tuple(chosen[:SET_B_SIZE])


def presentation_order(codes: Sequence[str], seed: int = SEED) -> dict[str, str]:
    """New neutral codes ``t01``... for ``codes``, in an order drawn with ``seed``."""
    order = np.random.default_rng(seed).permutation(len(codes))
    return {f"t{i + 1:02d}": codes[int(j)] for i, j in enumerate(order)}


def error(counts: Sequence[int], reference: Sequence[int]) -> int:
    return sum(abs(c - r) for c, r in zip(counts, reference, strict=True))


def recovered(
    automatic: Sequence[int], reviewed: Sequence[int], reference: Sequence[int]
) -> tuple[int, int]:
    """(passages the automatic counts are short of the reference, how many of them the
    reviewed counts make up), per direction and summed. Counting past the reference does
    not recover more."""
    shortfall = made_up = 0
    for a, v, r in zip(automatic, reviewed, reference, strict=True):
        short = max(r - a, 0)
        shortfall += short
        made_up += min(max(v - a, 0), short)
    return shortfall, made_up


def fragment_counts(changes: Iterable[Mapping[str, Any]]) -> int:
    """Count decisions on tracks that do not cross alone and sit next to a counted track."""
    return sum(
        c["verdict"] == "counted"
        and not c["crosses_line_alone"]
        and bool(c["counted_tracks_nearby"])
        for c in changes
    )


def answers(events: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """How often each answer to the same-fish question was recorded in review events."""
    by_reason = {reason: key for key, reason in SAME_FISH_REASONS.items()}
    found = Counter(by_reason[e["reason"]] for e in events if e.get("reason") in by_reason)
    return {key: found.get(key, 0) for key in SAME_FISH_REASONS}


def group_target(per_reviewer: Mapping[str, bool | None]) -> dict[str, Any]:
    """A target over the reviewers who took part: met when it holds for at least two thirds
    of them, with at least ``MIN_REVIEWERS``. A reviewer with no complete clip counts as not
    meeting it."""
    n = len(per_reviewer)
    met = sum(value is True for value in per_reviewer.values())
    needed = math.ceil(2 * n / 3)
    if n < MIN_REVIEWERS:
        outcome = f"not evaluable: {n} reviewer(s), at least {MIN_REVIEWERS} needed"
    else:
        outcome = "met" if met >= needed else "not met"
    return {"reviewers": n, "met_by": met, "needed": needed, "outcome": outcome}
