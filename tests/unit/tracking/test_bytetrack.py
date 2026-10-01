from __future__ import annotations

import numpy as np
import pytest

from passagewatch.detection.classical import FrameDetections
from passagewatch.tracking.bytetrack import ByteTrackConfig, ByteTracker


def box(x: float, y: float, w: float = 40.0, h: float = 12.0) -> list[float]:
    return [x, y, x + w, y + h]


def frame(*items: tuple[list[float], float]) -> FrameDetections:
    if not items:
        return FrameDetections(np.empty((0, 4)), np.empty(0))
    return FrameDetections(
        np.array([b for b, _ in items], dtype=np.float64),
        np.array([s for _, s in items], dtype=np.float64),
    )


def run(frames: list[FrameDetections], **overrides: object) -> list:  # type: ignore[type-arg]
    return ByteTracker(ByteTrackConfig.model_validate(overrides)).run(frames)


def test_a_steadily_detected_fish_is_one_trajectory() -> None:
    frames = [frame((box(10 + 4 * t, 50), 0.9)) for t in range(20)]

    tracks = run(frames)

    assert len(tracks) == 1
    assert tracks[0].frames.tolist() == list(range(20))


def test_low_score_detections_keep_a_fading_fish_on_its_track() -> None:
    # Frames 8-11: the fish is only detected with score 0.3, below high_threshold 0.5.
    frames = [frame((box(10 + 4 * t, 50), 0.3 if 8 <= t < 12 else 0.9)) for t in range(20)]

    tracks = run(frames, max_lost=1)
    high_only = run(frames, max_lost=1, low_threshold=0.5)

    assert len(tracks) == 1 and len(tracks[0].frames) == 20
    assert tracks[0].scores[8:12].tolist() == pytest.approx([0.3] * 4)
    # Without the second (low-score) stage the track breaks into two.
    assert len(high_only) == 2


def test_crossing_fish_at_different_depths_keep_their_tracks() -> None:
    frames = [frame((box(10 + 6 * t, 20), 0.9), (box(130 - 6 * t, 80), 0.9)) for t in range(20)]

    tracks = run(frames)

    assert len(tracks) == 2
    for track in tracks:
        assert len(set(track.boxes[:, 1].tolist())) == 1  # never switches fish


def test_low_score_detections_never_start_a_track() -> None:
    frames = [frame((box(10 + 4 * t, 50), 0.3)) for t in range(20)]

    assert run(frames) == []


def test_a_long_gap_ends_the_track_and_short_tracks_are_dropped() -> None:
    frames = [frame((box(10 + 4 * t, 50), 0.9)) if not 10 <= t < 20 else frame() for t in range(30)]

    split = run(frames, max_lost=5)
    bridged = run(frames, max_lost=12)
    filtered = run(frames[:2], min_length=3)

    assert [len(t.frames) for t in split] == [10, 10]
    assert len(bridged) == 1 and len(bridged[0].frames) == 20
    assert filtered == []


def test_frame_offset_is_applied() -> None:
    tracks = ByteTracker(ByteTrackConfig()).run(
        [frame((box(10 + 4 * t, 50), 0.9)) for t in range(5)], frame_offset=100
    )

    assert tracks[0].frames.tolist() == [100, 101, 102, 103, 104]


def test_inconsistent_thresholds_are_refused() -> None:
    with pytest.raises(ValueError, match="low_threshold"):
        ByteTracker(ByteTrackConfig(high_threshold=0.3, low_threshold=0.4))
