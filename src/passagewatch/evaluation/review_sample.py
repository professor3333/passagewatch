"""The independently reviewed development sample (Stage 9; docs/review_sample.md).

A reviewer other than the CFC annotators reviews the release's results on a sample of
kenai-holdout-v1 clips in the review interface. Their decisions are review-quality labels:
for each track the release proposed, whether its count contribution is right, and the fish
it missed. A second reviewer reviews part of the sample, so the labels' agreement is known.
Disagreements, with the CFC reference or between reviewers, are recorded, never resolved.

- **Selection.** A stratified random sample, proportional to the eligible clips per
  camera and reference-passage bin, at least one per stratum, by largest remainder.
- **Verdicts.** Each track ends in one of: ``kept`` (the reviewer confirmed the release),
  ``removed`` (a counted passage is not one), ``counted`` (an uncounted track is a
  passage), ``redirected`` (a passage in the other direction), ``unresolved``, or
  ``unreviewed`` (no decision).
- **Agreement.** Between two reviewers, on the tracks of the clips both reviewed (matched by
  clip and track ID): the share of tracks with the same final outcome (counted right,
  counted left, not counted, unresolved), and Cohen's kappa over those outcomes.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from passagewatch.evaluation.study_design import Candidate

SEED = 0
SAMPLE_SIZE = 50
SECOND_REVIEWER_CLIPS = 15
PASSAGE_BINS = ((0, 0), (1, 2), (3, 4), (5, 10_000))
VERDICTS = ("kept", "removed", "counted", "redirected", "unresolved", "unreviewed")


def stratum(c: Candidate) -> tuple[str, int]:
    index = next(i for i, (lo, hi) in enumerate(PASSAGE_BINS) if lo <= c.passages <= hi)
    return c.camera, index


def select(pool: list[Candidate], n: int = SAMPLE_SIZE, seed: int = SEED) -> list[Candidate]:
    """``n`` clips, stratified and proportional, in a fixed random presentation order."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, int], list[Candidate]] = {}
    for c in sorted(pool, key=lambda c: c.clip_name):
        strata.setdefault(stratum(c), []).append(c)
    keys = sorted(strata)
    if n < len(keys) or n > len(pool):
        raise ValueError(f"cannot take {n} clips from {len(pool)} in {len(keys)} strata")
    exact = {k: n * len(strata[k]) / len(pool) for k in keys}
    take = {k: max(1, math.floor(exact[k])) for k in keys}
    by_remainder = sorted(keys, key=lambda k: (-(exact[k] - math.floor(exact[k])), k))
    i = 0
    while sum(take.values()) < n:
        k = by_remainder[i % len(keys)]
        if take[k] < len(strata[k]):
            take[k] += 1
        i += 1
    while sum(take.values()) > n:  # the minimum of one per stratum overshot
        k = max((k for k in keys if take[k] > 1), key=lambda k: take[k] - exact[k])
        take[k] -= 1
    chosen = []
    for k in keys:
        members = list(strata[k])
        rng.shuffle(members)
        chosen += members[: take[k]]
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


@dataclass(frozen=True)
class TrackLabel:
    track_id: int
    triage: str | None
    automatic: str | None  # the release's direction; None: not counted
    final: str | None  # after review; None: not counted (or unresolved)
    state: str  # the API's review state: automatic, accepted, rejected, corrected, unresolved

    @property
    def verdict(self) -> str:
        if self.state == "automatic":
            return "unreviewed"
        if self.state == "unresolved":
            return "unresolved"
        if self.final == self.automatic:
            return "kept"
        if self.automatic is None:
            return "counted"
        if self.final is None:
            return "removed"
        return "redirected"

    @property
    def outcome(self) -> str:
        """The track's final count contribution, for agreement between reviewers."""
        if self.state == "unresolved":
            return "unresolved"
        return self.final or "none"


def track_labels(tracks: Iterable[dict[str, Any]]) -> list[TrackLabel]:
    """Labels from a job's ``/tracks`` rows (latest revision)."""
    return [
        TrackLabel(
            track_id=int(t["track_id"]),
            triage=t.get("triage"),
            automatic=t.get("direction"),
            final=t.get("final_direction"),
            state=str(t["review_state"]),
        )
        for t in tracks
    ]


def counts_error(counts: tuple[int, int], reference: tuple[int, int]) -> int:
    return abs(counts[0] - reference[0]) + abs(counts[1] - reference[1])


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa of two raters' labels of the same items; nan when undefined."""
    if len(a) != len(b) or not a:
        return math.nan
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return math.nan if expected == 1 else (observed - expected) / (1 - expected)


def clip_summary(
    labels: list[TrackLabel],
    reviewed: tuple[int, int],
    automatic: tuple[int, int],
    reference: tuple[int, int],
    added: int,
) -> dict[str, Any]:
    verdicts = Counter(label.verdict for label in labels)
    return {
        "reference": list(reference),
        "automatic": list(automatic),
        "reviewed": list(reviewed),
        "automatic_error": counts_error(automatic, reference),
        "reviewed_error": counts_error(reviewed, reference),
        "agrees_with_reference": reviewed == reference,
        "tracks": len(labels),
        "verdicts": {v: verdicts.get(v, 0) for v in VERDICTS},
        "added_passages": added,
    }


def triage_table(labels: list[TrackLabel]) -> dict[str, dict[str, Any]]:
    """Per triage state, how often the reviewer changed the release's decision."""
    table: dict[str, dict[str, Any]] = {}
    for triage in sorted({label.triage or "none" for label in labels}):
        group = [
            label
            for label in labels
            if (label.triage or "none") == triage and label.verdict != "unreviewed"
        ]
        changed = sum(label.verdict in ("removed", "counted", "redirected") for label in group)
        table[triage] = {
            "reviewed_tracks": len(group),
            "changed": changed,
            "unresolved": sum(label.verdict == "unresolved" for label in group),
            "changed_share": changed / len(group) if group else math.nan,
        }
    return table


def agreement(
    first: dict[str, list[TrackLabel]], second: dict[str, list[TrackLabel]]
) -> dict[str, Any]:
    """Agreement of two reviewers on the tracks both of them decided, matched by (clip,
    track ID). Tracks either reviewer left unreviewed are left out."""

    def outcomes(by_clip: dict[str, list[TrackLabel]]) -> dict[tuple[str, int], str]:
        return {
            (clip, t.track_id): t.outcome
            for clip, labels in by_clip.items()
            for t in labels
            if t.verdict != "unreviewed"
        }

    a, b = outcomes(first), outcomes(second)
    keys = sorted(set(a) & set(b))
    xs, ys = [a[k] for k in keys], [b[k] for k in keys]
    same = sum(x == y for x, y in zip(xs, ys, strict=True))
    return {
        "clips": sorted(set(first) & set(second)),
        "tracks": len(keys),
        "same_outcome": same,
        "share": same / len(keys) if keys else math.nan,
        "kappa": cohen_kappa(xs, ys),
        "disagreements": [
            {"clip": clip, "track_id": track, "first": a[(clip, track)], "second": b[(clip, track)]}
            for clip, track in keys
            if a[(clip, track)] != b[(clip, track)]
        ],
    }
