"""The review page's same-fish question, replayed on recorded reviews.

Before a reviewer counts a track the release did not count, the review page asks whether
it is a different fish from the tracks that currently count and overlap it in time, or
come within ``SAME_FISH_FRAMES`` of it (``countedNearby`` in frontend/src/review.ts; the
rule here must stay the same). In the usability study, the reviewed sample and the
documentation check, reviewers counted pieces of fish that another track already counted.

:func:`replay` goes through a job's review events in order and reports, for each such
"count" decision, which counted tracks the page would have named at that moment: the
tracks the release counted, changed by the decisions recorded before it. It describes
whether the question would have appeared; it cannot say how a reviewer would answer it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SAME_FISH_FRAMES = 10
# The ``reason`` the review page sends with each answer to the question
# (SAME_FISH_REASONS in frontend/src/review.ts).
SAME_FISH_REASONS = {
    "same": "same-fish question: same fish",
    "different": "same-fish question: different fish",
}


def counts_now(track: Mapping[str, Any], decisions: Mapping[str, Any]) -> bool:
    """Whether a track adds to the count under ``decisions`` (a revision's decisions)."""
    decision = decisions.get("tracks", {}).get(str(track["track_id"]))
    if decision is None:
        return track["direction"] is not None
    return bool(decision["state"] != "unresolved" and decision["direction"] is not None)


def counted_nearby(
    track: Mapping[str, Any],
    tracks: Iterable[Mapping[str, Any]],
    decisions: Mapping[str, Any],
    gap: int = SAME_FISH_FRAMES,
) -> list[int]:
    """IDs of the tracks that count and overlap ``track`` in time, or come within ``gap``."""
    return [
        int(other["track_id"])
        for other in tracks
        if other["track_id"] != track["track_id"]
        and counts_now(other, decisions)
        and other["start_frame"] <= track["end_frame"] + gap
        and track["start_frame"] <= other["end_frame"] + gap
    ]


@dataclass(frozen=True)
class CountDecision:
    track_id: int
    direction: str
    counted_nearby: list[int]

    @property
    def asked(self) -> bool:
        return bool(self.counted_nearby)


def replay(
    tracks: Sequence[Mapping[str, Any]],
    events: Iterable[tuple[int, Mapping[str, Any]]],
    decisions_at: Mapping[int, Mapping[str, Any]],
) -> list[CountDecision]:
    """The "count" decisions among ``events`` ((base revision, payload) in order), each with
    the counted tracks the page would have named; ``decisions_at`` maps a revision to its
    decisions. Redirecting a counted passage is not a count decision."""
    by_id = {int(t["track_id"]): t for t in tracks}
    found = []
    for base, event in events:
        if event.get("action") != "set_direction":
            continue
        track = by_id[int(event["track_id"])]
        if track["direction"] is not None:
            continue
        found.append(
            CountDecision(
                int(track["track_id"]),
                str(event["direction"]),
                counted_nearby(track, tracks, decisions_at[base]),
            )
        )
    return found


def question_rate(tracks: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """(tracks the release did not count, those of them that would show the question) in
    the automatic result, before any review."""
    uncounted = [t for t in tracks if t["direction"] is None]
    asked = sum(bool(counted_nearby(t, tracks, {})) for t in uncounted)
    return len(uncounted), asked
