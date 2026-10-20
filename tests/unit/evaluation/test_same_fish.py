from __future__ import annotations

from typing import Any

from passagewatch.evaluation.same_fish import counted_nearby, counts_now, question_rate, replay


def track(track_id: int, start: int, end: int, direction: str | None) -> dict[str, Any]:
    return {"track_id": track_id, "start_frame": start, "end_frame": end, "direction": direction}


TRACKS = [
    track(1, 0, 20, "right"),  # counted by the release
    track(2, 30, 40, None),  # starts 10 frames after track 1 ends
    track(3, 31, 50, None),  # 11 frames after track 1; overlaps track 2
    track(4, 100, 120, None),  # alone
]


def test_the_rule_matches_the_review_page() -> None:
    # The same cases as countedNearby's tests in frontend/src/review.test.ts.
    assert counted_nearby(TRACKS[1], TRACKS, {}) == [1]
    assert counted_nearby(TRACKS[2], TRACKS, {}) == []
    assert counted_nearby(TRACKS[3], TRACKS, {}) == []


def test_decisions_change_what_counts() -> None:
    rejected = {"tracks": {"1": {"state": "rejected", "direction": None}}}
    unresolved = {"tracks": {"1": {"state": "unresolved", "direction": "right"}}}
    piece_counted = {"tracks": {"2": {"state": "corrected", "direction": "right"}}}
    assert not counts_now(TRACKS[0], rejected) and not counts_now(TRACKS[0], unresolved)
    assert counted_nearby(TRACKS[2], TRACKS, piece_counted) == [2]


def test_replay_names_the_counted_tracks_at_each_count_decision() -> None:
    piece_counted = {"tracks": {"2": {"state": "corrected", "direction": "right"}}}
    events = [
        (0, {"action": "accept", "track_id": 1}),
        (1, {"action": "set_direction", "track_id": 2, "direction": "right"}),
        (2, {"action": "set_direction", "track_id": 3, "direction": "right"}),
        (3, {"action": "set_direction", "track_id": 1, "direction": "left"}),  # a redirect
        (4, {"action": "set_direction", "track_id": 4, "direction": "left"}),
    ]
    decisions = {0: {}, 1: {}, 2: piece_counted, 3: piece_counted, 4: piece_counted}
    found = replay(TRACKS, events, decisions)
    assert [(d.track_id, d.counted_nearby, d.asked) for d in found] == [
        (2, [1], True),
        (3, [2], True),  # the piece counted just before
        (4, [], False),
    ]


def test_question_rate_counts_uncounted_tracks_with_a_counted_neighbour() -> None:
    assert question_rate(TRACKS) == (3, 1)
