from __future__ import annotations

import numpy as np

from passagewatch.counting.policy import CFC_COMPATIBLE_V1
from passagewatch.evaluation.errors import analyze_clip, combine, recall_by_size
from passagewatch.ingestion.mot import BoxAnnotations

W, H = 200, 100


def path(track_id: int, frames: range, u0: float, u1: float, v: float = 0.5) -> list[tuple]:
    """A 10 x 10 px box moving linearly from u0 to u1 (normalized x) over ``frames``."""
    n = len(frames)
    rows = []
    for i, f in enumerate(frames):
        u = u0 + (u1 - u0) * (i / max(1, n - 1))
        rows.append((f, track_id, u * W - 5, v * H - 5, u * W + 5, v * H + 5))
    return rows


def ann(*paths: list[tuple]) -> BoxAnnotations:
    rows = [r for p in paths for r in p]
    if not rows:
        return BoxAnnotations.empty()
    a = np.asarray(rows, dtype=np.float64)
    return BoxAnnotations(a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2:])


def analyze(reference: BoxAnnotations, predicted: BoxAnnotations, **kwargs: object):  # type: ignore[no-untyped-def]
    return analyze_clip("c", reference, predicted, W, H, CFC_COMPATIBLE_V1, **kwargs)  # type: ignore[arg-type]


PASS_RIGHT = path(1, range(0, 21), 0.2, 0.8)


def test_a_followed_passage_is_counted() -> None:
    result = analyze(ann(PASS_RIGHT), ann(path(7, range(0, 21), 0.2, 0.8)))

    assert result.reference == {"counted": 1} and result.predicted == {"counted": 1}
    assert result.confusion == {("right", "right"): 1}


def test_a_fish_with_no_track_is_missed() -> None:
    result = analyze(ann(PASS_RIGHT), ann())

    assert result.reference == {"missed_fish": 1}
    assert result.confusion == {("right", "none"): 1}


def test_opposite_direction_is_wrong_direction() -> None:
    # The predicted track passes the reference fish going the other way.
    result = analyze(ann(PASS_RIGHT), ann(path(7, range(0, 21), 0.8, 0.2)), min_shared=1)

    assert result.reference == {"wrong_direction": 1} and result.predicted == {"wrong_direction": 1}


def test_fragments_that_miss_the_line_are_a_split_track() -> None:
    fragments = ann(path(7, range(0, 8), 0.2, 0.41), path(8, range(13, 21), 0.59, 0.8))

    assert analyze(ann(PASS_RIGHT), fragments).reference == {"split_track": 1}


def test_a_short_track_is_partial_or_ambiguous_near_the_line() -> None:
    partial = analyze(ann(PASS_RIGHT), ann(path(7, range(0, 7), 0.2, 0.38)))
    ambiguous = analyze(ann(PASS_RIGHT), ann(path(7, range(0, 10), 0.2, 0.47)))

    assert partial.reference == {"partial_track": 1}
    assert ambiguous.reference == {"ambiguous_start_end": 1}


def test_a_track_that_jumps_to_another_fish_is_merged() -> None:
    passing = path(1, range(0, 21), 0.2, 0.8, v=0.3)
    resting = path(2, range(0, 21), 0.3, 0.3, v=0.7)
    jumping = path(7, range(0, 6), 0.2, 0.35, v=0.3) + path(7, range(6, 21), 0.3, 0.3, v=0.7)

    result = analyze(ann(passing, resting), ann(jumping))

    assert result.reference == {"merged_track": 1}


def test_false_passages_are_background_or_a_non_passing_fish() -> None:
    clutter = path(7, range(0, 21), 0.2, 0.8, v=0.1)  # nowhere near a fish
    resting = path(2, range(0, 21), 0.48, 0.52, v=0.7)  # stationary reference fish
    crossing = path(8, range(0, 21), 0.44, 0.56, v=0.7)  # but its track crosses

    result = analyze(ann(resting), ann(clutter, crossing))

    assert result.predicted == {"background": 1, "non_passing_fish": 1}
    totals = combine([result])
    assert totals["predicted_passages"]["background"] == 1  # type: ignore[index]
    assert totals["direction_confusion"]["reference_none"]["predicted_right"] == 2  # type: ignore[index]


def test_recall_by_size_bins_reference_boxes_by_area() -> None:
    small = (0, 1, 10.0, 10.0, 14.0, 14.0)  # 16 px^2
    large = (0, 2, 50.0, 50.0, 70.0, 70.0)  # 400 px^2
    reference = ann([small, large])
    detected = [np.array([[50.0, 50.0, 70.0, 70.0]])]

    total, found = recall_by_size(reference, detected, 0, area_m2_per_px=1e-4, bins=(0.01,))

    assert (total[0], found[0]) == (1, 0)  # 0.0016 m^2: missed
    assert (total[1], found[1]) == (1, 1)  # 0.04 m^2: found


def test_a_fish_counted_by_two_tracks_is_a_duplicate() -> None:
    # Two overlapping predicted tracks both follow the same passing fish.
    first = path(7, range(0, 21), 0.2, 0.8)
    second = path(8, range(2, 21), 0.25, 0.8)

    result = analyze(ann(PASS_RIGHT), ann(first, second))

    assert result.reference == {"counted": 1}
    assert result.predicted == {"counted": 1, "duplicate": 1}
