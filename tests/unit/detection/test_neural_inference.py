from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from passagewatch.counting.policy import DirectionalCounts
from passagewatch.detection.classical import FrameDetections
from passagewatch.detection.neural import build_yolox
from passagewatch.inference.neural import (
    LoadedDetector,
    above,
    detect_frames,
    load_detections,
    load_detector,
    save_detections,
    track_and_evaluate,
)
from passagewatch.ingestion.cfc import Clip
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.mot import BoxAnnotations
from passagewatch.preprocessing.letterbox import PREPROCESSING_VERSION, InputSize
from passagewatch.tracking.kalman import TrackerConfig


def save_checkpoint(path: Path, preprocessing: str = PREPROCESSING_VERSION) -> None:
    torch.save(
        {
            "config": {"name": "t", "model_size": "tiny", "input_height": 96, "input_width": 64},
            "preprocessing_version": preprocessing,
            "model": build_yolox("tiny").state_dict(),
            "state": {"epoch": 7},
        },
        path,
    )


def test_checkpoint_loads_with_its_input_size(tmp_path: Path) -> None:
    save_checkpoint(tmp_path / "epoch-007.pt")

    detector = load_detector(tmp_path / "epoch-007.pt", torch.device("cpu"))

    assert detector.input_size == InputSize(96, 64)
    assert detector.epoch == 7 and len(detector.checkpoint_sha256) == 64
    assert not detector.model.training


def test_checkpoint_with_other_preprocessing_is_refused(tmp_path: Path) -> None:
    save_checkpoint(tmp_path / "old.pt", preprocessing="letterbox-gray3-v0")

    with pytest.raises(ValueError, match="trained with preprocessing"):
        load_detector(tmp_path / "old.pt", torch.device("cpu"))


class FixedOutputs(torch.nn.Module):
    """Stands in for YOLOX in eval mode: one confident box per image, in input pixels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], 2, 6)
        out[:, 0] = torch.tensor([30.0, 20.0, 20.0, 10.0, 0.9, 1.0])  # cx, cy, w, h, obj, cls
        out[:, 1] = torch.tensor([45.0, 90.0, 20.0, 20.0, 0.9, 1.0])  # extends past the frame
        return out


def test_detections_map_back_to_frame_pixels_and_are_clipped() -> None:
    detector = LoadedDetector(
        model=FixedOutputs(),  # type: ignore[arg-type]
        input_size=InputSize(96, 64),
        device=torch.device("cpu"),
        checkpoint_sha256="0" * 64,
        epoch=1,
        run_name="t",
    )
    frame = np.zeros((192, 100), dtype=np.uint8)  # letterbox scale = min(96/192, 64/100) = 0.5

    result = detect_frames(detector, [frame])[0]

    np.testing.assert_allclose(result.boxes[0], [40.0, 30.0, 80.0, 50.0])  # input box / 0.5
    np.testing.assert_allclose(result.boxes[1], [70.0, 160.0, 100.0, 192.0])  # clipped
    assert result.scores.tolist() == pytest.approx([0.9, 0.9])


def test_cached_detections_round_trip(tmp_path: Path) -> None:
    detections = [
        FrameDetections(np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([0.7])),
        FrameDetections(np.empty((0, 4)), np.empty(0)),
        FrameDetections(np.array([[5.0, 6, 7, 8], [9.0, 9, 11, 12]]), np.array([0.2, 0.9])),
    ]

    save_detections(tmp_path / "c.npz", 12, detections)
    start, loaded = load_detections(tmp_path / "c.npz")

    assert start == 12 and len(loaded) == 3
    for a, b in zip(detections, loaded, strict=True):
        np.testing.assert_array_equal(a.boxes, b.boxes)
        np.testing.assert_array_equal(a.scores, b.scores)
    assert [len(d.scores) for d in above(loaded, 0.5)] == [1, 0, 1]


def test_reference_boxes_as_detections_reproduce_the_reference_counts() -> None:
    meta = ClipMetadata(
        clip_name="c_2018-06-01_120000_0_30",
        num_frames=30,
        framerate=10.0,
        width=200,
        height=100,
        x_meter_start=0.0,
        x_meter_stop=2.0,
        y_meter_start=1.0,
        y_meter_stop=0.0,
    )
    frames = np.arange(30)
    x = 10.0 + 5.0 * frames  # one fish crossing left to right
    reference = BoxAnnotations(
        frame_index=frames,
        track_id=np.ones(30, dtype=np.int64),
        boxes=np.stack([x, np.full(30, 40.0), x + 20, np.full(30, 50.0)], axis=1),
    )
    clip = Clip("kenai-val", meta, 0, 30, reference, None)
    detections = [FrameDetections(reference.boxes[i : i + 1], np.ones(1)) for i in range(30)]

    evaluation = track_and_evaluate(clip, detections, TrackerConfig())

    assert evaluation.error.reference == evaluation.error.predicted == DirectionalCounts(right=1)
    assert evaluation.detection.recall == 1.0 and evaluation.tracks == 1
