from __future__ import annotations

import numpy as np

from passagewatch.detection.classical import DetectorConfig, FrameDetections, Scale, detect_clip
from passagewatch.evaluation.detection import match_detections, pairwise_iou
from passagewatch.ingestion.mot import BoxAnnotations


def test_pairwise_iou() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])

    np.testing.assert_allclose(pairwise_iou(a, b), [[1.0, 1 / 3, 0.0]])


def test_matching_is_one_to_one_per_frame() -> None:
    reference = BoxAnnotations(
        frame_index=np.array([5, 5, 6]),
        track_id=np.array([1, 2, 1]),
        boxes=np.array([[0, 0, 10, 10], [50, 50, 60, 60], [2, 0, 12, 10]], dtype=np.float64),
    )
    detections = [
        # Frame 5: two detections on the first fish (only one can match), none on the second.
        FrameDetections(np.array([[0, 0, 10, 10], [1, 0, 11, 10]], dtype=np.float64), np.ones(2)),
        # Frame 6: one good match.
        FrameDetections(np.array([[2, 0, 12, 10]], dtype=np.float64), np.ones(1)),
    ]

    match = match_detections(reference, detections, frame_start=5)

    assert (match.matched, match.reference_boxes, match.detections) == (2, 3, 3)
    assert match.recall == 2 / 3 and match.precision == 2 / 3


def test_range_dependent_noise_bands_find_a_dim_near_range_fish() -> None:
    # Most of the frame (far range, top rows) is very noisy; the near range (bottom 20 rows)
    # is quiet. A fish with a contrast of 30 at near range is below 3 x the frame-wide
    # noise, but far above 3 x the near-range noise.
    rng = np.random.default_rng(1)
    n, h, w = 30, 120, 100
    frames = np.empty((n, h, w), dtype=np.float32)
    frames[:, :100] = rng.normal(100, 40, (n, 100, w))
    frames[:, 100:] = rng.normal(60, 3, (n, 20, w))
    for t in range(n):
        frames[t, 106:114, 10 + 2 * t : 30 + 2 * t] += 30
    frames_u8 = np.clip(frames, 0, 255).astype(np.uint8)
    scale = Scale(sx=1.0, sy=1.0, meters_per_px_x=0.01, meters_per_px_y=0.01)

    def near_range_hits(config: DetectorConfig) -> int:
        return sum(any(b[1] >= 100 for b in d.boxes) for d in detect_clip(frames_u8, scale, config))

    banded = near_range_hits(DetectorConfig(noise_bands=6, min_contrast=0))
    single = near_range_hits(DetectorConfig(noise_bands=1, min_contrast=0))
    assert banded >= 28
    assert single <= banded // 2
