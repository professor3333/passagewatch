from __future__ import annotations

import math

import pytest

from passagewatch.evaluation.usability import (
    Trial,
    analyze,
    group_result,
    participant_result,
    review_actions,
    sus_score,
    trials_from_rows,
)


def trial(
    participant: str, condition: str, clip: str, seconds: float, final: tuple[int, int]
) -> Trial:
    return Trial(participant, condition, clip, seconds, final, (2, 1), (2, 0))


def person(code: str, manual_s: float, assisted_s: float, assisted_final=(2, 1)) -> list[Trial]:
    return [trial(code, "manual", f"m{i}", manual_s, (2, 1)) for i in range(6)] + [
        trial(code, "assisted", f"a{i}", assisted_s, assisted_final) for i in range(6)
    ]


def test_rows_skip_practice_and_excluded_clips() -> None:
    plan = {"clips": {k: {"reference": [1, 0], "automatic": [1, 1]} for k in ("p01", "c01", "c02")}}
    base = {"participant": "D1", "condition": "manual", "active_ms": "30000"}
    rows = [
        base | {"clip": "p01", "practice": "1", "final_right": "1", "final_left": "0"},
        base | {"clip": "c01", "practice": "0", "final_right": "1", "final_left": "0"},
        base | {"clip": "c02", "practice": "0", "final_right": "0", "final_left": "0"},
    ]

    (kept,) = trials_from_rows(rows, plan, excluded={("D1", "c02")})
    assert (kept.clip, kept.active_s, kept.error, kept.automatic) == ("c01", 30.0, 0, (1, 1))


def test_a_participant_who_is_faster_and_as_accurate_meets_the_target() -> None:
    result = participant_result(person("D1", manual_s=100, assisted_s=60))

    assert result["time_saving"] == pytest.approx(0.4)
    assert result["time_saving_ci"] == pytest.approx([0.4, 0.4])  # constant times
    assert result["nmae"] == {"manual": 0.0, "assisted": 0.0}
    assert result["automatic_nmae_on_assisted_clips"] == pytest.approx(1 / 3)
    assert (result["errors_fixed"], result["errors_introduced"]) == (6, 0)
    assert result["target_met"]


def test_faster_but_less_accurate_does_not_meet_the_target() -> None:
    result = participant_result(person("P1", 100, 50, assisted_final=(2, 0)))

    assert result["nmae_difference"] == pytest.approx(1 / 3)
    assert not result["target_met"]


def test_group_results_need_three_independent_participants() -> None:
    two = person("P3", 100, 60) + person("P1", 100, 90)
    three = two + person("P2", 100, 75)

    assert group_result(two) is None
    group = group_result(three)
    assert group is not None
    ratio = math.exp((math.log(0.6) + math.log(0.9) + math.log(0.75)) / 3)
    assert group["time_ratio_geometric_mean"] == pytest.approx(ratio)
    assert group["time_saving_ci"][0] <= 1 - ratio <= group["time_saving_ci"][1]


def test_questionnaires_and_roles_are_reported() -> None:
    forms = [{"participant": "D1", "condition": "manual", "sus": [3] * 10, "tlx": [40] * 6}]
    result = analyze(person("D1", 100, 60), forms)

    assert result["participants"]["D1"]["role"] == "developer"
    assert result["participants"]["D1"]["questionnaires"] == {
        "manual": {"sus": 50.0, "tlx_raw": 40.0}
    }
    assert sus_score([5, 1, 5, 1, 5, 1, 5, 1, 5, 1]) == 100


def test_the_developer_never_counts_towards_the_group() -> None:
    # Only the developer is faster; the two independent participants are not.
    developer_and_two = person("D1", 100, 20) + person("P1", 100, 100) + person("P2", 100, 110)
    three_independent = person("P1", 100, 90) + person("P2", 100, 75) + person("P3", 100, 60)

    assert group_result(developer_and_two) is None
    group = group_result(three_independent + person("D1", 100, 1))
    assert group is not None and group["participants"] == ["P1", "P2", "P3"]
    ratio = math.exp((math.log(0.9) + math.log(0.75) + math.log(0.6)) / 3)
    assert group["time_ratio_geometric_mean"] == pytest.approx(ratio)


def test_an_excluded_clip_does_not_bias_the_time_ratio() -> None:
    # Same pace in both conditions; one manual clip excluded leaves 5 against 6.
    trials = person("P1", 100, 100)
    trials.remove(next(t for t in trials if t.condition == "manual"))

    result = participant_result(trials)
    assert result["clips"] == {"manual": 5, "assisted": 6}
    assert result["mean_active_s"] == {"manual": 100.0, "assisted": 100.0}
    assert result["time_saving"] == pytest.approx(0.0)


def test_review_actions_count_tracks_turned_into_passages() -> None:
    def row(action: str, automatic: str, practice: str = "0") -> dict[str, str]:
        return {
            "participant": "P1",
            "practice": practice,
            "action": action,
            "automatic_direction": automatic,
        }

    rows = [
        row("accept", "right"),
        row("set_direction", "right"),
        row("set_direction", ""),
        row("set_direction", "", practice="1"),
        row("add_passage", ""),
    ]

    assert review_actions(rows) == {
        "P1": {
            "by_action": {"accept": 1, "set_direction": 2, "add_passage": 1},
            "set_direction_on_non_passages": 1,
        }
    }
