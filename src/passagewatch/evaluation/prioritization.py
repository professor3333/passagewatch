"""How quickly a review queue leads a reviewer to the counting errors.

Every predicted trajectory is a queue item. Reviewing it takes ``overhead_s`` plus its
duration, and reveals:

- itself, if it is a **false passage** (a predicted passage that the one-to-one error
  analysis does not count: duplicate, background, non-passing fish, wrong direction);
- every **missed reference passage** it follows (split, merged, partial, ambiguous or
  wrong-direction cases), since the reviewer then sees the fish and can add the passage.

A reference passage with no predicted trajectory at all (``missed_fish``) cannot be reached
from the queue; only random audits can find it. It counts in the denominator of "all errors"
but not of "reachable errors".

``found_curve`` walks the queue in a given order and reports the share of errors found
when a given share of the total review time has been spent. The design target is 80% of
errors within 40% of the review time (``docs/design.md``, a planning target).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from passagewatch.evaluation.errors import ClipErrors


@dataclass(frozen=True)
class QueueItem:
    clip_name: str
    track_id: int
    cost_s: float
    reveals: frozenset[str]  # error IDs found by reviewing this trajectory


@dataclass(frozen=True)
class ErrorSet:
    reachable: frozenset[str]
    unreachable: frozenset[str]  # missed fish with no trajectory at all

    @property
    def total(self) -> int:
        return len(self.reachable) + len(self.unreachable)


def queue_items(
    analysis: ClipErrors,
    spans_frames: dict[int, int],
    framerate: float,
    *,
    overhead_s: float = 3.0,
) -> tuple[list[QueueItem], ErrorSet]:
    """The queue items and errors of one clip. ``spans_frames`` maps every predicted
    trajectory to its span in frames."""
    clip = analysis.clip_name
    missed = {r for r, outcome in analysis.reference_outcomes.items() if outcome != "counted"}
    items = []
    reachable: set[str] = set()
    for track_id, span in sorted(spans_frames.items()):
        follows = analysis.follows.get(track_id, frozenset())
        reveals = {f"{clip}:missed:{r}" for r in follows & missed}
        if analysis.predicted_outcomes.get(track_id, "counted") != "counted":
            reveals.add(f"{clip}:false:{track_id}")
        reachable |= reveals
        items.append(QueueItem(clip, track_id, overhead_s + span / framerate, frozenset(reveals)))
    all_missed = {f"{clip}:missed:{r}" for r in missed}
    return items, ErrorSet(frozenset(reachable), frozenset(all_missed - reachable))


def found_curve(
    ordered: Sequence[QueueItem], errors: ErrorSet, budgets: Sequence[float]
) -> dict[float, tuple[float, float]]:
    """For each review-time share in ``budgets``: (share of all errors found, share of
    reachable errors found), reviewing ``ordered`` from the front."""
    total_cost = sum(i.cost_s for i in ordered)
    cumulative = np.cumsum([i.cost_s for i in ordered]) if ordered else np.array([])
    result = {}
    for budget in budgets:
        n = int(np.searchsorted(cumulative, budget * total_cost, side="right"))
        found = set().union(*(i.reveals for i in ordered[:n])) if n else set()
        result[budget] = (
            len(found) / errors.total if errors.total else 1.0,
            len(found) / len(errors.reachable) if errors.reachable else 1.0,
        )
    return result


def random_order_curve(
    items: Sequence[QueueItem],
    errors: ErrorSet,
    budgets: Sequence[float],
    *,
    repeats: int = 200,
    seed: int = 0,
) -> dict[float, tuple[float, float]]:
    """The same curve averaged over random review orders: the baseline to beat."""
    rng = np.random.default_rng(seed)
    sums = {b: np.zeros(2) for b in budgets}
    for _ in range(repeats):
        order = [items[i] for i in rng.permutation(len(items))]
        for b, found in found_curve(order, errors, budgets).items():
            sums[b] += found
    return {b: (float(v[0] / repeats), float(v[1] / repeats)) for b, v in sums.items()}


def ordered(items: Sequence[QueueItem], key: Callable[[QueueItem], object]) -> list[QueueItem]:
    return sorted(items, key=lambda i: (key(i), i.clip_name, i.track_id))
