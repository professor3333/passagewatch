from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from passagewatch.ingestion.cfc import CfcLayout, load_clip
from passagewatch.ingestion.splits import Split, official_split
from passagewatch.validation.alignment import box_contrast, contrast_by_offset
from passagewatch.visualization.overlay import (
    box_pixel_corners,
    contact_sheet,
    draw_frame,
    write_video,
)

from .conftest import LOCATION, FakeCfc, clip_name, mot_line

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_box_outline_covers_exactly_its_pixels() -> None:
    # Internal box [2, 6) x [3, 8) covers pixel columns 2..5 and rows 3..7.
    assert box_pixel_corners([2.0, 3.0, 6.0, 8.0]) == ((2, 3), (5, 7))
    # Fractional edges round outward to the pixels they touch.
    assert box_pixel_corners([2.5, 3.2, 5.5, 7.9]) == ((2, 3), (5, 7))

    # Box [20, 26) x [30, 38): columns 20..25, rows 30..37. The track label sits above row 30.
    box = np.array([[20.0, 30.0, 26.0, 38.0]])
    lit = draw_frame(np.zeros((60, 60), dtype=np.uint8), box, [1]).any(axis=2)
    assert lit[30, 20:26].all() and lit[37, 20:26].all()
    assert lit[30:38, 20].all() and lit[30:38, 25].all()
    assert not lit[31:37, 21:25].any()
    assert not lit[30:, :20].any() and not lit[30:, 26:].any() and not lit[38:].any()


def test_counting_line_is_drawn_at_its_column() -> None:
    canvas = draw_frame(np.zeros((10, 20), dtype=np.uint8), np.empty((0, 4)), [], line_x=10.0)

    assert np.argwhere(canvas.any(axis=2))[:, 1].tolist() == [10] * 10


def test_box_contrast_is_positive_on_a_bright_target() -> None:
    image = np.full((40, 40), 50, dtype=np.float32)
    image[10:20, 10:20] = 200

    assert box_contrast(image, np.array([10.0, 10.0, 20.0, 20.0])) == pytest.approx(150.0)
    assert box_contrast(image, np.array([50.0, 50.0, 60.0, 60.0])) is None


def write_moving_target(fake: FakeCfc) -> str:
    """A clip whose bright 6x6 target moves 4 px right per frame, annotated 1-based."""
    name = clip_name("moving")
    fake.add_clip(name, [mot_line(f, 1, 1 + 4 * (f - 1) % 30, 20, 6, 6) for f in range(11, 21)])
    frames = fake.tiny / "raw" / LOCATION / name
    for i in range(10, 20):
        image = np.full((60, 40), 40, dtype=np.uint8)
        x = 4 * i % 30
        image[20:26, x : x + 6] = 220
        cv2.imwrite(str(frames / f"{i}.jpg"), image)
    return name


def test_contrast_peaks_at_offset_zero_for_aligned_annotations(fake_cfc: FakeCfc) -> None:
    name = write_moving_target(fake_cfc)
    layout = CfcLayout.tiny(fake_cfc.root)
    clip = load_clip(layout, LOCATION, layout.metadata(LOCATION).clips[name])

    contrast = contrast_by_offset(clip, offsets=[-1, 0, 1])

    assert max(contrast, key=lambda k: contrast[k]) == 0


def test_viewer_writes_a_sheet_and_a_video(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    name = write_moving_target(fake_cfc)
    layout = CfcLayout.tiny(fake_cfc.root)
    clip = load_clip(layout, LOCATION, layout.metadata(LOCATION).clips[name])

    sheet = contact_sheet(clip, count=4, columns=2, tile_height=60)
    written = write_video(clip, tmp_path / "clip.mp4")

    assert sheet.shape[0] == 120
    assert written == 10
    capture = cv2.VideoCapture(str(tmp_path / "clip.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    assert math.isclose(capture.get(cv2.CAP_PROP_FPS), 10.0)


TINY_ROOT = REPO_ROOT / "data/extracted/cfc"


@pytest.mark.slow
@pytest.mark.skipif(
    not (TINY_ROOT / "tiny_dataset/tiny_dataset").is_dir(), reason="CFC tiny subset not present"
)
def test_real_tiny_boxes_align_with_their_frames() -> None:
    """Regression test for the 1-based -> 0-based frame conversion, on train/val clips only."""
    layout = CfcLayout.tiny(TINY_ROOT)
    offsets = [-1, 0, 1]
    totals = dict.fromkeys(offsets, 0.0)
    best_at_zero = clips = 0
    for location in ("kenai-train", "kenai-val"):
        assert official_split(location) is not Split.TEST
        for meta in layout.metadata(location).clips.values():
            contrast = contrast_by_offset(load_clip(layout, location, meta), offsets)
            if any(math.isnan(v) for v in contrast.values()):
                continue
            clips += 1
            best_at_zero += max(contrast, key=lambda k: contrast[k]) == 0
            for k in offsets:
                totals[k] += contrast[k]

    assert clips == 40
    assert totals[0] > totals[1] and totals[0] > totals[-1]
    assert best_at_zero >= 30
