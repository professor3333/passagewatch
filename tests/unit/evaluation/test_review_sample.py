from __future__ import annotations

import math
from collections import Counter

import pytest

from passagewatch.evaluation.review_sample import (
    TrackLabel,
    agreement,
    clip_summary,
    cohen_kappa,
    describe_changes,
    review_timing,
    select,
    stratum,
    track_paths,
    triage_table,
)
from passagewatch.evaluation.study_design import Candidate


def candidate(i: int, camera: str, passages: int) -> Candidate:
    return Candidate(f"c{i:03d}_{camera}_x", 300, 10.0, (passages, 0), (passages, 0))


def pool() -> list[Candidate]:
    clips = [candidate(i, "LeftFar", 1) for i in range(60)]
    clips += [candidate(100 + i, "LeftNear", 6) for i in range(30)]
    clips += [candidate(200 + i, "LeftFar", 0) for i in range(3)]
    return clips


def test_selection_is_proportional_reproducible_and_covers_every_stratum() -> None:
    chosen = select(pool(), n=31)

    assert len(chosen) == 31 and len({c.clip_name for c in chosen}) == 31
    per = Counter(stratum(c) for c in chosen)
    assert per == {("LeftFar", 1): 20, ("LeftNear", 3): 10, ("LeftFar", 0): 1}
    assert [c.clip_name for c in select(pool(), n=31)] == [c.clip_name for c in chosen]


def test_selection_refuses_impossible_sizes() -> None:
    with pytest.raises(ValueError):
        select(pool(), n=2)  # three strata


def label(track: int, automatic: str | None, final: str | None, state: str) -> TrackLabel:
    return TrackLabel(track, "suggested", automatic, final, state)


def test_verdicts_describe_the_change_to_the_count() -> None:
    assert label(1, "right", "right", "accepted").verdict == "kept"
    assert label(2, None, None, "rejected").verdict == "kept"  # not a fish, not counted
    assert label(3, "right", None, "rejected").verdict == "removed"
    assert label(4, None, "left", "corrected").verdict == "counted"
    assert label(5, "right", "left", "corrected").verdict == "redirected"
    assert label(6, "right", None, "unresolved").verdict == "unresolved"
    assert label(7, "right", "right", "automatic").verdict == "unreviewed"


def test_clip_summary_compares_with_the_reference() -> None:
    labels = [label(1, "right", "right", "accepted"), label(2, None, "right", "corrected")]
    summary = clip_summary(labels, reviewed=(3, 0), automatic=(1, 0), reference=(3, 0), added=1)

    assert summary["automatic_error"] == 2 and summary["reviewed_error"] == 0
    assert summary["agrees_with_reference"]
    assert summary["verdicts"]["counted"] == 1 and summary["added_passages"] == 1


def test_triage_table_counts_changed_decisions() -> None:
    labels = [
        TrackLabel(1, "needs_review", "right", None, "rejected"),
        TrackLabel(2, "needs_review", "right", "right", "accepted"),
        TrackLabel(3, "suggested", "right", "right", "accepted"),
        TrackLabel(4, "suggested", "right", "right", "automatic"),
    ]
    table = triage_table(labels)

    assert table["needs_review"] == {
        "reviewed_tracks": 2,
        "changed": 1,
        "unresolved": 0,
        "changed_share": 0.5,
    }
    assert table["suggested"]["reviewed_tracks"] == 1  # unreviewed tracks are left out


def test_agreement_matches_tracks_by_clip_and_id() -> None:
    first = {
        "s01": [label(1, "right", "right", "accepted"), label(2, None, None, "accepted")],
        "s02": [label(1, "right", "right", "accepted")],
    }
    second = {
        "s01": [label(1, "right", "right", "accepted"), label(2, None, "left", "corrected")],
        "s03": [label(1, "right", None, "rejected")],
    }
    result = agreement(first, second)

    assert result["clips"] == ["s01"] and result["tracks"] == 2 and result["same_outcome"] == 1
    assert result["disagreements"] == [
        {"clip": "s01", "track_id": 2, "first": "none", "second": "left"}
    ]


def test_cohen_kappa() -> None:
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert math.isnan(cohen_kappa(["a", "a"], ["a", "a"]))  # no variation: undefined


def test_agreement_leaves_out_tracks_either_reviewer_did_not_decide() -> None:
    first = {"s01": [label(1, "right", "right", "accepted"), label(2, None, None, "automatic")]}
    second = {"s01": [label(1, "right", "right", "automatic"), label(2, None, None, "automatic")]}

    result = agreement(first, second)
    assert result["tracks"] == 0 and math.isnan(result["share"])


def box(track: int, frame: int, x_min: float, x_max: float) -> dict[str, float]:
    return {"track_id": track, "frame_index": frame, "x_min": x_min, "x_max": x_max}


def test_track_paths_use_the_first_and_last_box_centers() -> None:
    boxes = [box(1, 5, 80, 100), box(1, 2, 10, 30), box(1, 9, 60, 80)]
    path = track_paths(boxes, width=200)[1]
    assert (path.start_frame, path.end_frame) == (2, 9)
    assert (path.start_x, path.end_x) == (0.1, 0.35)


def test_changes_say_whether_a_track_crosses_and_which_counted_tracks_are_near() -> None:
    labels = [
        TrackLabel(1, "suggested", "right", "right", "accepted"),  # counted, frames 0-20
        TrackLabel(2, "suggested", None, "right", "corrected"),  # a piece after track 1
        TrackLabel(3, "suggested", None, "left", "corrected"),  # crosses on its own, later
        TrackLabel(4, "suggested", None, None, "accepted"),  # unchanged
    ]
    boxes = [box(1, 0, 20, 40), box(1, 20, 110, 130)]
    boxes += [box(2, 25, 130, 150), box(2, 40, 170, 190)]
    boxes += [box(3, 100, 150, 170), box(3, 120, 30, 50)]
    boxes += [box(4, 0, 0, 10), box(4, 5, 0, 10)]
    changes = describe_changes(labels, track_paths(boxes, width=200), line_x=0.5)
    assert [c["track_id"] for c in changes] == [2, 3]
    assert changes[0]["crosses_line_alone"] is False
    assert changes[0]["counted_tracks_nearby"] == [1]  # 5 frames after track 1 ends
    assert changes[1]["crosses_line_alone"] is True
    assert changes[1]["counted_tracks_nearby"] == []


def test_review_timing_spans_the_first_to_the_last_decision() -> None:
    timing = review_timing(
        {
            "s02": ["2026-10-17T10:05:00+00:00", "2026-10-17T10:06:30+00:00"],
            "s01": ["2026-10-17T10:00:00+00:00"],
            "s03": [],
        }
    )
    assert list(timing["clips"]) == ["s01", "s02"]
    assert timing["clips"]["s02"]["events"] == 2
    assert timing["span_minutes"] == 6.5
    assert review_timing({})["span_minutes"] is None
