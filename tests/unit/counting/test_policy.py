from __future__ import annotations

import numpy as np
import pytest

from passagewatch.counting.policy import (
    CFC_COMPATIBLE_V1,
    CountingPolicy,
    Direction,
    DirectionalCounts,
    Outcome,
    RiverCounts,
    count_trajectories,
    is_benchmark_compatible,
    tally,
    to_river_directions,
)
from passagewatch.ingestion.mot import BoxAnnotations

W, H = 200, 100


def track(track_id: int, path: list[tuple[int, float]], v: float = 0.5) -> list[tuple]:
    """Boxes of width 10 and height 10 whose centers are at normalized (u, v) per frame."""
    return [(frame, track_id, u * W - 5, v * H - 5, u * W + 5, v * H + 5) for frame, u in path]


def annotations(*tracks: list[tuple]) -> BoxAnnotations:
    rows = [r for t in tracks for r in t]
    if not rows:
        return BoxAnnotations.empty()
    arr = np.asarray(rows, dtype=np.float64)
    return BoxAnnotations(
        frame_index=arr[:, 0].astype(np.int64),
        track_id=arr[:, 1].astype(np.int64),
        boxes=arr[:, 2:],
    )


def counts(*tracks: list[tuple]) -> DirectionalCounts:
    return tally(count_trajectories(annotations(*tracks), W, H))


# The counting fixtures required by the project's testing priorities.


def test_left_to_right_passage_counts_one_right() -> None:
    assert counts(track(1, [(0, 0.2), (5, 0.4), (10, 0.8)])) == DirectionalCounts(right=1)


def test_right_to_left_passage_counts_one_left() -> None:
    assert counts(track(1, [(0, 0.9), (5, 0.6), (10, 0.1)])) == DirectionalCounts(left=1)


def test_stationary_fish_counts_nothing() -> None:
    result = count_trajectories(annotations(track(1, [(0, 0.49), (30, 0.51)])), W, H)

    assert result[0].outcome is Outcome.STATIONARY
    assert tally(result) == DirectionalCounts()


def test_fish_that_approaches_and_retreats_counts_nothing() -> None:
    result = count_trajectories(annotations(track(1, [(0, 0.1), (5, 0.48), (10, 0.2)])), W, H)

    assert result[0].outcome is Outcome.NO_CROSSING
    assert tally(result) == DirectionalCounts()


def test_fish_that_crosses_and_returns_counts_zero_not_two() -> None:
    assert counts(track(1, [(0, 0.2), (5, 0.8), (10, 0.3)])) == DirectionalCounts()


def test_missing_frame_gap_does_not_change_the_count() -> None:
    gappy = track(1, [(0, 0.2), (1, 0.25), (20, 0.75), (21, 0.8)])

    assert counts(gappy) == DirectionalCounts(right=1)


def test_track_split_near_the_line_loses_the_passage() -> None:
    # One fish, two fragments: 0.30 -> 0.49 and 0.51 -> 0.80. Neither spans the line.
    assert counts(track(1, [(0, 0.3), (5, 0.49)]), track(2, [(6, 0.51), (10, 0.8)])) == (
        DirectionalCounts()
    )
    # If one fragment spans the line, the passage survives.
    assert counts(track(1, [(0, 0.3), (5, 0.52)]), track(2, [(6, 0.55), (10, 0.8)])) == (
        DirectionalCounts(right=1)
    )


# Edge cases of the rule.


def test_line_position_uses_greater_or_equal_on_the_right() -> None:
    assert counts(track(1, [(0, 0.3), (5, 0.5)])) == DirectionalCounts(right=1)
    assert counts(track(1, [(0, 0.5), (5, 0.3)])) == DirectionalCounts(left=1)
    assert counts(track(1, [(0, 0.5), (5, 0.9)])) == DirectionalCounts()


def test_single_observation_is_stationary() -> None:
    result = count_trajectories(annotations(track(1, [(3, 0.2)])), W, H)

    assert result[0].outcome is Outcome.STATIONARY
    assert result[0].observations == 1


def test_stationary_filter_is_anisotropic_like_the_official_code() -> None:
    # 9 px horizontally on a 200 px wide frame is 0.045 < 0.05: stationary, even though it
    # crosses the line. 9 px vertically on a 100 px tall frame is 0.09: not stationary.
    horizontal = track(1, [(0, 0.4775), (5, 0.5225)])
    assert counts(horizontal) == DirectionalCounts()
    result = count_trajectories(
        annotations([(0, 1, 95.0, 10.0, 105.0, 20.0), (5, 1, 95.0, 19.0, 105.0, 29.0)]), W, H
    )
    assert result[0].outcome is Outcome.NO_CROSSING


def test_displacement_at_threshold_is_not_stationary() -> None:
    policy = CountingPolicy(version="test", min_displacement_normalized=0.25)
    result = count_trajectories(annotations(track(1, [(0, 0.375), (5, 0.625)])), W, H, policy)

    assert result[0].outcome is Outcome.PASSAGE


def test_observation_order_in_the_input_does_not_matter() -> None:
    shuffled = track(1, [(10, 0.8), (0, 0.2), (5, 0.6)])

    result = count_trajectories(annotations(shuffled), W, H)

    assert (result[0].start_frame, result[0].end_frame) == (0, 10)
    assert result[0].direction is Direction.RIGHT


def test_two_boxes_in_one_frame_is_an_error() -> None:
    with pytest.raises(ValueError, match="more than one box"):
        count_trajectories(annotations(track(1, [(0, 0.2), (0, 0.3), (5, 0.8)])), W, H)


def test_each_trajectory_counts_at_most_once_and_ids_are_per_track() -> None:
    result = counts(
        track(1, [(0, 0.1), (10, 0.9)]),
        track(2, [(0, 0.2), (10, 0.7)]),
        track(7, [(0, 0.95), (10, 0.05)]),
        track(9, [(0, 0.1), (10, 0.3)]),
    )

    assert result == DirectionalCounts(right=2, left=1)


def test_empty_annotations_count_nothing() -> None:
    assert counts() == DirectionalCounts()


def test_custom_line_position() -> None:
    policy = CountingPolicy(version="test", line_x_normalized=0.25)
    result = count_trajectories(annotations(track(1, [(0, 0.1), (5, 0.4)])), W, H, policy)

    assert result[0].direction is Direction.RIGHT
    assert not is_benchmark_compatible(policy)
    assert is_benchmark_compatible(CFC_COMPATIBLE_V1)


@pytest.mark.parametrize("line", [0.0, 1.0, -0.1])
def test_line_must_be_inside_the_frame(line: float) -> None:
    with pytest.raises(ValueError, match="line_x_normalized"):
        CountingPolicy(version="test", line_x_normalized=line)


# Direction mapping.


def test_river_directions_need_configured_orientation() -> None:
    image = DirectionalCounts(right=7, left=2)

    assert to_river_directions(image, None) is None
    assert to_river_directions(image, Direction.RIGHT) == RiverCounts(upstream=7, downstream=2)
    assert to_river_directions(image, Direction.LEFT) == RiverCounts(upstream=2, downstream=7)
    assert RiverCounts(upstream=7, downstream=2).net == 5


def test_horizontal_flip_swaps_directions() -> None:
    path = [(0, 0.2), (5, 0.45), (10, 0.85)]
    flipped = [(f, 1.0 - u) for f, u in path]

    assert counts(track(1, path)) == DirectionalCounts(right=1)
    assert counts(track(1, flipped)) == DirectionalCounts(left=1)
