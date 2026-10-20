from __future__ import annotations

from passagewatch.evaluation.retest import (
    answers,
    fragment_counts,
    group_target,
    presentation_order,
    recovered,
    set_b,
)


def test_set_b_takes_the_first_undercounted_clips_in_code_order() -> None:
    clips = {f"s{i:02d}": {"reference": [1, 0], "automatic": [1, 0]} for i in range(1, 51)}
    for code in ("s10", "s17", "s20", "s21", "s30", "s31", "s40", "s45"):
        clips[code] = {"reference": [2, 1], "automatic": [2, 0]}
    clips["s18"] = {"reference": [1, 0], "automatic": [2, 0]}  # overcounted: not chosen
    assert set_b(clips) == ("s17", "s20", "s21", "s30", "s31", "s40")


def test_the_presentation_order_is_fixed_by_the_seed() -> None:
    codes = [f"s{i:02d}" for i in range(1, 22)]
    order = presentation_order(codes)
    assert sorted(order) == [f"t{i:02d}" for i in range(1, 22)]
    assert sorted(order.values()) == codes
    assert order == presentation_order(codes)
    assert list(order.values()) != codes


def test_recovered_counts_only_up_to_the_shortfall() -> None:
    assert recovered((1, 0), (1, 0), (3, 1)) == (3, 0)
    assert recovered((1, 0), (2, 1), (3, 1)) == (3, 2)
    assert recovered((1, 0), (5, 0), (3, 1)) == (3, 2)  # overcounting recovers no more
    assert recovered((2, 0), (1, 0), (2, 0)) == (0, 0)


def test_fragment_counts_follow_the_reviewed_sample_definition() -> None:
    changes = [
        {"verdict": "counted", "crosses_line_alone": False, "counted_tracks_nearby": [3]},
        {"verdict": "counted", "crosses_line_alone": False, "counted_tracks_nearby": []},
        {"verdict": "counted", "crosses_line_alone": True, "counted_tracks_nearby": [3]},
        {"verdict": "redirected", "crosses_line_alone": False, "counted_tracks_nearby": [3]},
    ]
    assert fragment_counts(changes) == 1


def test_answers_are_read_from_the_reason() -> None:
    events = [
        {"action": "accept", "reason": "same-fish question: same fish"},
        {"action": "accept", "reason": ""},
        {"action": "set_direction", "reason": "same-fish question: different fish"},
        {"action": "accept", "reason": "same-fish question: same fish"},
    ]
    assert answers(events) == {"same": 2, "different": 1}


def test_a_target_needs_two_thirds_of_at_least_three_reviewers() -> None:
    assert group_target({"T1": True, "T2": True, "T3": False})["outcome"] == "met"
    assert group_target({"T1": True, "T2": False, "T3": None})["outcome"] == "not met"
    assert group_target({"T1": True, "T2": True, "T3": True, "T4": False})["outcome"] == "met"
    assert group_target({"T1": True, "T2": True, "T3": False, "T4": False})["outcome"] == "not met"
    assert group_target({"T1": True, "T2": True})["outcome"].startswith("not evaluable")
