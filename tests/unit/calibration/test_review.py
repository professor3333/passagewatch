from __future__ import annotations

import itertools

import numpy as np
import pytest

from passagewatch.calibration.audit import audit_windows, seed_from, unflagged_mask
from passagewatch.calibration.review_score import (
    ReviewConfig,
    TrackFeatures,
    review,
    track_features,
)
from passagewatch.counting.policy import CFC_COMPATIBLE_V1, count_trajectories
from passagewatch.tracking.kalman import Trajectory, trajectories_to_annotations

W, H = 200, 100
M_PER_PX = (0.01, 0.01)


def trajectory(
    track_id: int, xs: list[float], *, y: float = 50.0, score: float = 0.8, start: int = 0
) -> Trajectory:
    frames = np.arange(start, start + len(xs))
    boxes = np.array([[x - 5, y - 5, x + 5, y + 5] for x in xs], dtype=np.float64)
    return Trajectory(track_id, frames, boxes, np.full(len(xs), score))


def features(*trajectories: Trajectory) -> dict[int, TrackFeatures]:
    counts = count_trajectories(trajectories_to_annotations(list(trajectories)), W, H)
    found = track_features(trajectories, counts, M_PER_PX, CFC_COMPATIBLE_V1.line_x_normalized)
    return {f.track_id: f for f in found}


def test_a_clean_passage_is_suggested() -> None:
    f = features(trajectory(1, list(np.linspace(20, 180, 20))))[1]

    r = review(f)

    assert f.is_passage and f.observations == 20 and f.gap_fraction == 0.0
    assert f.straightness == pytest.approx(1.0) and f.crowding == 0.0
    assert r.triage == "suggested" and r.review_score > 0.9 and r.reasons == ()


def test_weak_short_passages_need_review() -> None:
    f = features(trajectory(1, [60.0, 80.0, 120.0, 140.0], score=0.25))[1]

    r = review(f)

    assert r.triage == "needs_review"
    assert {"detection", "duration"} <= set(r.reasons)


def test_back_and_forth_or_on_the_line_passages_are_unresolved() -> None:
    zigzag = [40.0, 160.0, 40.0, 160.0, 40.0, 160.0]  # ends right of the line: a passage
    on_line = list(np.linspace(20, 100.5, 12))  # ends 0.0025 past the line

    f = features(trajectory(1, zigzag), trajectory(2, on_line, y=90.0))

    assert review(f[1]).triage == "unresolved" and review(f[1]).reasons[0] == "back_and_forth"
    assert review(f[2]).triage == "unresolved" and review(f[2]).reasons[0] == "endpoint_on_line"


def test_a_fragment_ending_near_the_line_is_a_possible_missed_passage() -> None:
    fragment = list(np.linspace(20, 95, 10))  # stops 0.025 short of the line
    far = list(np.linspace(20, 50, 10))

    f = features(trajectory(1, fragment), trajectory(2, far, y=90.0))

    assert not f[1].is_passage
    assert review(f[1]).triage == "needs_review"
    assert review(f[1]).reasons[0] == "possible_missed_passage"
    assert review(f[2]).triage == "suggested"


def test_overlapping_tracks_count_as_crowded_and_gaps_as_gaps() -> None:
    a = trajectory(1, list(np.linspace(20, 180, 10)))
    b = trajectory(2, list(np.linspace(22, 182, 10)))
    gappy = Trajectory(
        3, np.array([0, 1, 5, 9]), a.boxes[:4] + np.array([0.0, 40.0, 0.0, 40.0]), np.full(4, 0.8)
    )

    f = features(a, b, gappy)

    assert f[1].crowding == 1.0 and f[2].crowding == 1.0
    assert f[3].gap_fraction == pytest.approx(1 - 4 / 10)


def test_the_suggest_threshold_is_configurable() -> None:
    f = features(trajectory(1, list(np.linspace(20, 180, 8)), score=0.4))[1]

    assert review(f, ReviewConfig(suggest_threshold=0.1)).triage == "suggested"
    assert review(f, ReviewConfig(suggest_threshold=0.99)).triage == "needs_review"


# -- audit ------------------------------------------------------------------------------


def test_flagged_frames_and_margins_are_excluded() -> None:
    mask = unflagged_mask(20, [(5, 7), (15, 30)], margin_frames=1)

    assert np.flatnonzero(~mask).tolist() == [4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19]


def test_audit_windows_avoid_flagged_footage_and_reach_the_fraction() -> None:
    flagged = [(100, 180), (400, 420)]

    windows = audit_windows(1000, flagged, window_frames=50, fraction=0.2, seed=seed_from("job-1"))

    mask = unflagged_mask(1000, flagged, 0)
    assert sum(w.frames for w in windows) >= 0.2 * mask.sum()
    for w in windows:
        assert mask[w.start_frame : w.stop_frame].all() and w.frames == 50
    for a, b in itertools.pairwise(windows):
        assert a.stop_frame <= b.start_frame
    assert windows == audit_windows(
        1000, flagged, window_frames=50, fraction=0.2, seed=seed_from("job-1")
    )
    assert windows != audit_windows(
        1000, flagged, window_frames=50, fraction=0.2, seed=seed_from("job-2")
    )


def test_short_unflagged_footage_gets_its_longest_stretch() -> None:
    windows = audit_windows(100, [(10, 60), (80, 90)], window_frames=50, fraction=0.5, seed=1)

    assert [(w.start_frame, w.stop_frame) for w in windows] == [(61, 80)]
    assert audit_windows(30, [(0, 29)], window_frames=10, fraction=0.5, seed=1) == []
