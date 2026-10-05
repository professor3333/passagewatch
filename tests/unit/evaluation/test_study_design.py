from __future__ import annotations

import pytest

from passagewatch.evaluation.study_design import (
    Candidate,
    assign_codes,
    candidates_from_report,
    eligible,
    latin_square,
    select,
)
from passagewatch.ingestion.metadata import ClipMetadata


def clip(
    i: int, camera: str, passages: int, frames: int = 200, rate: float = 9.0, error: int = 0
) -> Candidate:
    name = f"2018-05-27-JD147_{camera}_Stratum1_Set1_LO_2018-05-27_0{i:05d}_0_{frames}"
    return Candidate(name, frames, rate, (passages, 0), (passages + error, 0))


def pool() -> list[Candidate]:
    clips = []
    i = 0
    for camera in ("LeftFar", "LeftNear"):
        for passages in (1, 2, 3, 4, 5, 6):
            for error in (0, 1):
                clips.append(clip(i, camera, passages, error=error))
                i += 1
    return clips


def test_eligibility_uses_frames_duration_passages_and_documentation() -> None:
    ok = clip(1, "LeftFar", 3)
    clips = [
        ok,
        clip(2, "LeftFar", 3, frames=100),  # too few frames
        clip(3, "LeftFar", 3, rate=3.3),  # 200 frames but 61 s
        clip(4, "LeftFar", 0),  # no passage
        clip(5, "LeftFar", 9),  # too many
        clip(6, "LeftFar", 3),
    ]

    assert eligible(clips, documented={clips[-1].clip_name}) == [ok]


def test_sets_take_one_clip_per_stratum_and_are_balanced() -> None:
    practice, s1, s2 = select(pool(), seed=0)

    assert len(s1) == len(s2) == 6
    assert {c.stratum for c in s1} == {c.stratum for c in s2}  # one of every stratum each
    assert abs(sum(c.passages for c in s1) - sum(c.passages for c in s2)) <= 2
    assert abs(sum(c.automatic_error for c in s1) - sum(c.automatic_error for c in s2)) <= 2
    assert sorted(c.camera for c in practice) == ["LeftFar", "LeftNear"]
    names = [c.clip_name for c in (*practice, *s1, *s2)]
    assert len(set(names)) == 14  # no clip is used twice
    assert select(pool(), seed=0) == (practice, s1, s2)  # reproducible


def test_too_few_clips_in_a_stratum_are_refused() -> None:
    with pytest.raises(ValueError, match="two eligible clips"):
        select(pool()[:-3], seed=0)


def test_the_latin_square_balances_order_and_sets() -> None:
    plan = latin_square(["D1", "P1", "P2", "P3", "P4"])

    firsts = [(b[0]["condition"], b[0]["set"]) for b in list(plan.values())[:4]]
    assert sorted(firsts) == sorted(
        [("manual", "S1"), ("assisted", "S1"), ("manual", "S2"), ("assisted", "S2")]
    )
    for blocks in plan.values():
        assert {b["condition"] for b in blocks} == {"manual", "assisted"}
        assert {b["set"] for b in blocks} == {"S1", "S2"}
    assert plan["P4"] == plan["D1"]


def test_codes_hide_sets_and_keep_every_clip() -> None:
    practice, s1, s2 = select(pool())
    clips, practice_keys, sets = assign_codes(practice, s1, s2)

    assert practice_keys == ["p01", "p02"]
    assert [clips[k] for k in practice_keys] == practice
    assert sorted(sets["S1"] + sets["S2"]) == [f"c{i:02d}" for i in range(1, 13)]
    assert {clips[k].clip_name for k in sets["S1"]} == {c.clip_name for c in s1}
    assert assign_codes(practice, s1, s2) == (clips, practice_keys, sets)


def test_candidates_come_from_the_reports_selected_operating_point() -> None:
    name = "2018-05-27-JD147_LeftFar_Stratum1_Set1_LO_2018-05-27_000001_0_300"
    meta = ClipMetadata(
        clip_name=name,
        num_frames=300,
        framerate=10.0,
        width=100,
        height=200,
        x_meter_start=-1.0,
        x_meter_stop=1.0,
        y_meter_start=1.0,
        y_meter_stop=10.0,
    )

    def result(threshold: float, predicted: list[int]) -> dict[str, object]:
        clips = [{"clip_name": name, "reference": [2, 1], "predicted": predicted}]
        return {"epoch": 25, "threshold": threshold, "clips": clips}

    report = {
        "best": {"epoch": 25, "threshold": 0.4},
        "results": [result(0.3, [9, 9]), result(0.4, [2, 0])],
    }

    (c,) = candidates_from_report(report, {name: meta})
    assert (c.num_frames, c.seconds, c.reference, c.automatic) == (300, 30.0, (2, 1), (2, 0))
