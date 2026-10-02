from __future__ import annotations

import pytest

from passagewatch.evaluation.errors import ClipErrors
from passagewatch.evaluation.prioritization import (
    ErrorSet,
    QueueItem,
    found_curve,
    ordered,
    queue_items,
    random_order_curve,
)


def analysis() -> ClipErrors:
    """Reference passages 1 (counted by track 10), 2 (split over tracks 11 and 12, missed)
    and 3 (no track at all). Track 13 is a duplicate of 1; track 14 is background."""
    result = ClipErrors("c")
    result.reference_outcomes = {1: "counted", 2: "split_track", 3: "missed_fish"}
    result.predicted_outcomes = {10: "counted", 13: "duplicate", 14: "background"}
    result.follows = {
        10: frozenset({1}),
        11: frozenset({2}),
        12: frozenset({2}),
        13: frozenset({1}),
        14: frozenset(),
    }
    return result


def test_each_track_reveals_its_own_false_count_and_the_missed_fish_it_follows() -> None:
    items, errors = queue_items(analysis(), {10: 20, 11: 10, 12: 10, 13: 20, 14: 40}, 10.0)

    reveals = {i.track_id: i.reveals for i in items}
    assert reveals[10] == frozenset()
    assert reveals[11] == reveals[12] == {"c:missed:2"}
    assert reveals[13] == {"c:false:13"} and reveals[14] == {"c:false:14"}
    assert errors.reachable == {"c:missed:2", "c:false:13", "c:false:14"}
    assert errors.unreachable == {"c:missed:3"} and errors.total == 4
    assert {i.track_id: i.cost_s for i in items}[14] == pytest.approx(3.0 + 4.0)


def test_found_curve_counts_errors_within_the_review_time_share() -> None:
    errors = ErrorSet(frozenset({"a", "b"}), frozenset({"c"}))
    queue = [
        QueueItem("c", 1, 1.0, frozenset({"a"})),
        QueueItem("c", 2, 1.0, frozenset()),
        QueueItem("c", 3, 2.0, frozenset({"a", "b"})),
    ]

    curve = found_curve(queue, errors, [0.25, 0.5, 1.0])

    assert curve[0.25] == (pytest.approx(1 / 3), 0.5)  # first item: "a"
    assert curve[0.5] == (pytest.approx(1 / 3), 0.5)  # an empty second item
    assert curve[1.0] == (pytest.approx(2 / 3), 1.0)  # "b"; "c" is unreachable


def test_a_good_order_beats_random_order() -> None:
    errors = ErrorSet(frozenset({"e"}), frozenset())
    items = [QueueItem("c", i, 1.0, frozenset({"e"}) if i == 7 else frozenset()) for i in range(10)]

    best = found_curve(ordered(items, lambda i: i.track_id != 7), errors, [0.1])
    chance = random_order_curve(items, errors, [0.1], repeats=500)

    assert best[0.1] == (1.0, 1.0)
    assert chance[0.1][1] == pytest.approx(0.1, abs=0.05)
