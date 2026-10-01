"""Draw annotations on frames, and replay a clip as a video or a contact sheet.

Boxes are in the internal convention (``docs/counting_policy.md`` §1): continuous 0-based
coordinates where pixel column ``i`` covers ``[i, i+1)``. A box ``[x_min, x_max)`` therefore
covers pixel columns ``floor(x_min)`` to ``ceil(x_max) - 1``, which is where its outline is
drawn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from passagewatch.ingestion.cfc import Clip, frame_path

Image = NDArray[np.uint8]

# Distinct BGR colors; a track always gets the same one.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (75, 25, 230),
    (75, 180, 60),
    (25, 225, 255),
    (200, 130, 0),
    (48, 130, 245),
    (180, 30, 145),
    (240, 240, 70),
    (230, 50, 240),
    (60, 245, 210),
    (190, 190, 250),
)
LINE_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)


def track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def box_pixel_corners(box: Sequence[float]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Inclusive pixel corners of an internal-convention box."""
    x_min, y_min, x_max, y_max = box
    top_left = (math.floor(x_min), math.floor(y_min))
    bottom_right = (math.ceil(x_max) - 1, math.ceil(y_max) - 1)
    return top_left, bottom_right


def draw_frame(
    gray: NDArray[np.uint8],
    boxes: NDArray[np.float64],
    track_ids: Sequence[int],
    *,
    line_x: float | None = None,
    caption: str | None = None,
) -> Image:
    """Return a BGR copy of ``gray`` with boxes, track IDs, the line, and a caption."""
    canvas: Image = np.asarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    if line_x is not None:
        x = math.floor(line_x)
        cv2.line(canvas, (x, 0), (x, canvas.shape[0] - 1), LINE_COLOR, 1)
    for box, track in zip(boxes, track_ids, strict=True):
        color = track_color(int(track))
        top_left, bottom_right = box_pixel_corners(box.tolist())
        cv2.rectangle(canvas, top_left, bottom_right, color, 1)
        label_y = max(10, top_left[1] - 3)
        cv2.putText(
            canvas, str(int(track)), (top_left[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )
    if caption:
        cv2.putText(canvas, caption, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    return canvas


def render_clip_frame(clip: Clip, index: int, *, line_x_normalized: float | None = 0.5) -> Image:
    """Frame ``index`` (0-based) of ``clip`` with its boxes drawn."""
    if clip.frame_dir is None:
        raise ValueError(f"clip {clip.name} has no frames")
    decoded = cv2.imread(str(frame_path(clip.frame_dir, index)), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise ValueError(f"cannot decode frame {index} of {clip.name}")
    gray = np.asarray(decoded, dtype=np.uint8)
    mask = clip.annotations.frame_index == index
    line_x = None
    if line_x_normalized is not None:
        line_x = line_x_normalized * clip.metadata.width
    seconds = index / clip.metadata.framerate
    return draw_frame(
        gray,
        clip.annotations.boxes[mask],
        clip.annotations.track_id[mask].tolist(),
        line_x=line_x,
        caption=f"frame {index}  t={seconds:.1f}s  boxes {int(mask.sum())}",
    )


def _resize_to_height(image: Image, height: int) -> Image:
    scale = height / image.shape[0]
    width = max(1, round(image.shape[1] * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def contact_sheet(clip: Clip, *, count: int = 8, columns: int = 8, tile_height: int = 480) -> Image:
    """Evenly spaced frames of the clip window, side by side."""
    window = range(clip.frame_start, clip.frame_stop)
    count = min(count, len(window))
    picks = [window[round(i * (len(window) - 1) / max(1, count - 1))] for i in range(count)]
    tiles = [_resize_to_height(render_clip_frame(clip, i), tile_height) for i in picks]
    tile_width = max(t.shape[1] for t in tiles)
    rows = math.ceil(len(tiles) / columns)
    sheet: Image = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for n, tile in enumerate(tiles):
        r, c = divmod(n, columns)
        sheet[
            r * tile_height : (r + 1) * tile_height, c * tile_width : c * tile_width + tile.shape[1]
        ] = tile
    return sheet


def write_video(clip: Clip, path: Path, *, fps: float | None = None) -> int:
    """Write the clip window as an MP4 at the clip's frame rate; return frames written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first = render_clip_frame(clip, clip.frame_start)
    height, width = first.shape[:2]
    rate = fps or clip.metadata.framerate
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, rate, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {path}")
    written = 0
    try:
        for index in range(clip.frame_start, clip.frame_stop):
            frame = first if index == clip.frame_start else render_clip_frame(clip, index)
            if frame.shape[:2] != (height, width):
                raise ValueError(f"frame {index} of {clip.name} changes size")
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    return written
