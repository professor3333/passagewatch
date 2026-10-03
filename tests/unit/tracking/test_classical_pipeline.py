from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from passagewatch.counting.policy import CFC_COMPATIBLE_V1, DirectionalCounts
from passagewatch.detection.classical import (
    DetectorConfig,
    FrameDetections,
    Scale,
    analysis_size,
    detect_clip,
)
from passagewatch.evaluation.nmae import count_clip
from passagewatch.inference.classical import ClassicalConfig, load_classical_config, run_clip
from passagewatch.ingestion.cfc import Clip
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.mot import BoxAnnotations
from passagewatch.tracking.kalman import KalmanTracker, TrackerConfig, trajectories_to_annotations

REPO_ROOT = Path(__file__).resolve().parents[3]
H, W = 100, 160
# 1 cm per pixel: a 20 x 8 px blob is 0.016 m², inside the default size filter.
SCALE = Scale(sx=1.0, sy=1.0, meters_per_px_x=0.01, meters_per_px_y=0.01)


def synthetic_frames(
    n: int = 40, *, blobs: list[tuple[int, int, int, int, int]] | None = None, seed: int = 0
) -> tuple[np.ndarray, list[list[tuple[float, float, float, float]]]]:
    """Speckle-noise frames with bright moving blobs ``(x0, y, dx, w, h)``; returns truth boxes."""
    rng = np.random.default_rng(seed)
    blobs = [(10, 40, 3, 20, 8)] if blobs is None else blobs
    frames = np.clip(rng.normal(60, 6, (n, H, W)), 0, 255).astype(np.uint8)
    truth: list[list[tuple[float, float, float, float]]] = []
    for t in range(n):
        boxes = []
        for x0, y, dx, w, h in blobs:
            x = x0 + dx * t
            if x >= 0 and x + w <= W:
                frames[t, y : y + h, x : x + w] = 160
                boxes.append((float(x), float(y), float(x + w), float(y + h)))
        truth.append(boxes)
    return frames, truth


def iou(a: np.ndarray, b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def test_detector_finds_a_moving_bright_target_and_little_else() -> None:
    frames, truth = synthetic_frames()

    detections = detect_clip(frames, SCALE, DetectorConfig())

    hits = sum(
        any(iou(box, t) > 0.5 for box in det.boxes)
        for det, boxes in zip(detections, truth, strict=True)
        for t in boxes
    )
    false = sum(
        sum(all(iou(box, t) <= 0.1 for t in boxes) for box in det.boxes)
        for det, boxes in zip(detections, truth, strict=True)
    )
    assert hits >= 0.9 * sum(len(b) for b in truth)
    assert false <= 2


def test_size_filter_is_in_square_meters() -> None:
    frames, truth = synthetic_frames(blobs=[(10, 40, 3, 20, 8)])

    def blob_hits(detections: list[FrameDetections]) -> int:
        return sum(
            any(iou(box, t) > 0.3 for box in d.boxes)
            for d, boxes in zip(detections, truth, strict=True)
            for t in boxes
        )

    tiny_fish = DetectorConfig(min_area_m2=0.05)  # 500 px at 1 cm/px: bigger than the blob
    coarse = Scale(sx=1.0, sy=1.0, meters_per_px_x=0.1, meters_per_px_y=0.1)

    assert blob_hits(detect_clip(frames, SCALE, tiny_fish)) == 0
    # The same 160 px blob is 1.6 m² at 10 cm/px: above the default maximum of 0.6 m².
    assert blob_hits(detect_clip(frames, coarse, DetectorConfig())) == 0
    assert blob_hits(detect_clip(frames, SCALE, DetectorConfig())) > 30


def test_analysis_size_caps_height_and_keeps_aspect() -> None:
    assert analysis_size(789, 1933, 960) == (392, 960)
    assert analysis_size(288, 624, 960) == (288, 624)


def det(boxes: list[tuple[float, float, float, float]]) -> FrameDetections:
    return FrameDetections(np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.ones(len(boxes)))


def centers(frame: FrameDetections) -> np.ndarray:
    b = frame.boxes
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1) * 0.01


def track(frames: list[FrameDetections], config: TrackerConfig | None = None) -> BoxAnnotations:
    tracker = KalmanTracker(config or TrackerConfig())
    return trajectories_to_annotations(tracker.run(frames, [centers(f) for f in frames]))


def box_at(x: float, y: float) -> tuple[float, float, float, float]:
    return (x, y, x + 20, y + 8)


def test_two_fish_moving_in_opposite_directions_stay_separate() -> None:
    frames = [det([box_at(10 + 3 * t, 20), box_at(130 - 3 * t, 70)]) for t in range(30)]

    tracks = track(frames)

    assert sorted(np.unique(tracks.track_id).tolist()) == [1, 2]
    for tid in (1, 2):
        ys = tracks.boxes[tracks.track_id == tid, 1]
        assert len(set(ys.tolist())) == 1  # never swaps to the other fish


def test_short_gaps_are_bridged_and_long_gaps_split() -> None:
    def with_gap(gap: int) -> list[FrameDetections]:
        return [det([] if 10 <= t < 10 + gap else [box_at(5 + 3 * t, 40)]) for t in range(30)]

    bridged = track(with_gap(2), TrackerConfig(max_age=3))
    split = track(with_gap(6), TrackerConfig(max_age=3))

    assert len(np.unique(bridged.track_id)) == 1
    assert len(bridged) == 28  # observations are real detections only, never predictions
    assert len(np.unique(split.track_id)) == 2


def test_an_uncertainty_gate_stops_established_tracks_jumping_to_a_neighbor() -> None:
    # Fish A is tracked for 20 frames, then lost; fish B swims on 0.3 m away (within the
    # 0.5 m distance gate). Without the uncertainty gate, A's track takes over B. With the
    # default noise settings an established track's innovation SD settles near 0.12 m per
    # axis, so 2 SD is about 0.24 m.
    frames = [det([box_at(10 + 3 * t, 20)]) for t in range(20)]
    frames += [det([box_at(10 + 3 * t, 50)]) for t in range(20, 30)]

    plain = track(frames, TrackerConfig(gate_m=0.5))
    gated = track(frames, TrackerConfig(gate_m=0.5, gate_sigma=2.0))

    def owners(tracks: BoxAnnotations, y: float) -> set[int]:
        return set(tracks.track_id[tracks.boxes[:, 1] == y].tolist())

    assert owners(plain, 20.0) == owners(plain, 50.0)  # one track jumped between fish
    assert owners(gated, 20.0).isdisjoint(owners(gated, 50.0))


def test_an_uncertainty_gate_still_lets_new_tracks_catch_fast_fish() -> None:
    # 0.3 m per frame from the first frame: a new track's velocity is still uncertain.
    frames = [det([box_at(10 + 30 * t, 40)]) for t in range(12)]

    tracks = track(frames, TrackerConfig(gate_m=0.5, gate_sigma=2.0))

    assert np.unique(tracks.track_id).tolist() == [1] and len(tracks) == 12


def test_short_tracks_are_dropped() -> None:
    frames = [det([box_at(5 + 3 * t, 40)] if t < 2 else []) for t in range(10)]

    assert len(track(frames, TrackerConfig(min_length=3, min_hits=3))) == 0


def test_far_detections_are_not_associated() -> None:
    # A jump of 1 m in one frame is beyond the 0.5 m gate: two tracks, not one.
    frames = [det([box_at(10, 40)]) for _ in range(5)] + [det([box_at(110, 40)]) for _ in range(5)]

    assert len(np.unique(track(frames).track_id)) == 2


def test_repository_config_is_valid_and_hashed() -> None:
    config = load_classical_config(REPO_ROOT / "configs/tracking/classical-v1.yaml")

    assert config.name == "classical-v1"
    assert config.sha256() == config.model_copy().sha256()
    assert config.sha256() != config.model_copy(update={"name": "classical-v2"}).sha256()


def test_pipeline_counts_a_synthetic_passage(tmp_path: Path) -> None:
    frames, _ = synthetic_frames(n=40, blobs=[(5, 40, 3, 20, 8)])
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"{i}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta = ClipMetadata(
        clip_name="synthetic_2018-06-01_120000_0_40",
        num_frames=40,
        framerate=10.0,
        width=W,
        height=H,
        x_meter_start=0.0,
        x_meter_stop=W * 0.01,
        y_meter_start=H * 0.01,
        y_meter_stop=0.0,
    )
    clip = Clip("kenai-train", meta, 0, 40, BoxAnnotations.empty(), frame_dir)

    result = run_clip(clip, ClassicalConfig(name="test"))

    assert count_clip(result.tracks, meta, CFC_COMPATIBLE_V1) == DirectionalCounts(right=1)
    assert len(result.trajectories) == 1
    assert set(result.seconds) == {"decode", "detect", "track"}


@pytest.mark.parametrize("bad", [{"name": "Bad Name"}, {"name": "x", "extra": 1}])
def test_config_is_strict(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ClassicalConfig.model_validate(bad)


def test_tuned_config_keeps_its_recorded_values() -> None:
    # classical-v2 is the result of tuning plan 1; a different value needs a new name.
    v1 = load_classical_config(REPO_ROOT / "configs/tracking/classical-v1.yaml")
    v2 = load_classical_config(REPO_ROOT / "configs/tracking/classical-v2.yaml")

    assert v2.name == "classical-v2"
    assert (v2.detector.threshold_sigma, v2.detector.noise_bands) == (5.0, 4)
    assert (v2.tracker.max_age, v2.tracker.min_length) == (4, 8)
    unchanged = v2.model_copy(
        update={
            "name": v1.name,
            "detector": v2.detector.model_copy(update={"threshold_sigma": 3.0, "noise_bands": 1}),
            "tracker": v2.tracker.model_copy(update={"max_age": 3, "min_length": 3}),
        }
    )
    assert unchanged.sha256() == v1.sha256()
