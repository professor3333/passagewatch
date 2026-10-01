"""Paired bootstrap confidence intervals for directional nMAE, resampling whole clips.

Clips (or recording groups) are the resampling unit, never frames: frames within a clip are
strongly correlated. Two systems are compared on the **same** resampled clips, so the
interval of their difference accounts for the clips being shared.

nMAE is a ratio (summed absolute error over summed true passages); each resample recomputes
the ratio. Resamples whose clips contain no true passage are skipped and counted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class PairedComparison:
    a: Interval
    b: Interval
    difference: Interval  # b - a; negative means b has the lower error
    probability_b_better: float
    resamples: int
    skipped: int


def _errors(reference: NDArray[np.int64], predicted: NDArray[np.int64]) -> NDArray[np.int64]:
    """Per-clip nMAE numerators; arrays are ``(clips, 2)`` right/left counts."""
    return np.abs(predicted - reference).sum(axis=1)


def paired_bootstrap(
    reference: Sequence[Sequence[int]],
    predicted_a: Sequence[Sequence[int]],
    predicted_b: Sequence[Sequence[int]],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Percentile intervals for nMAE of systems a and b and their difference (b - a)."""
    ref = np.asarray(reference, dtype=np.int64)
    err_a = _errors(ref, np.asarray(predicted_a, dtype=np.int64))
    err_b = _errors(ref, np.asarray(predicted_b, dtype=np.int64))
    true = ref.sum(axis=1)
    if true.sum() == 0:
        raise ValueError("nMAE is undefined: no true passages")
    n = len(ref)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    denominators = true[idx].sum(axis=1)
    valid = denominators > 0
    nmae_a = err_a[idx].sum(axis=1)[valid] / denominators[valid]
    nmae_b = err_b[idx].sum(axis=1)[valid] / denominators[valid]
    diff = nmae_b - nmae_a
    tail = (1 - confidence) / 2 * 100

    def interval(point: float, samples: NDArray[np.float64]) -> Interval:
        low, high = np.percentile(samples, [tail, 100 - tail])
        return Interval(point, float(low), float(high))

    point_a = float(err_a.sum() / true.sum())
    point_b = float(err_b.sum() / true.sum())
    return PairedComparison(
        a=interval(point_a, nmae_a),
        b=interval(point_b, nmae_b),
        difference=interval(point_b - point_a, diff),
        probability_b_better=float((diff < 0).mean()),
        resamples=int(valid.sum()),
        skipped=int((~valid).sum()),
    )
