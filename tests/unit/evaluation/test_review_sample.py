from __future__ import annotations

import math
from collections import Counter

import pytest

from passagewatch.evaluation.review_sample import (
    TrackLabel,
    agreement,
    clip_summary,
    cohen_kappa,
    select,
    stratum,
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
