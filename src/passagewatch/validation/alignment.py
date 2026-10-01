"""An image-based check that boxes are drawn on the right frames.

Fish return brighter echoes than the water around them. If annotations are aligned with
the frames, a box is, on average, brighter than a ring around it, and this contrast is
highest when the box is compared with its own frame (offset 0) rather than a neighbor.
Treating CFC's 1-based frame numbers as 0-based would move the peak to offset +1.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from passagewatch.ingestion.cfc import Clip, frame_path


def box_contrast(image: NDArray[np.float32], box: NDArray[np.float64]) -> float | None:
    """Mean intensity inside ``box`` minus the mean of a surrounding ring.

    The ring extends half the box's larger side (at least 3 px) beyond the box, clipped to
    the image. Returns None when the box has no pixels inside the image.
    """
    h, w = image.shape
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    pad = max(3, (max(x1 - x0, y1 - y0)) // 2)
    inner = image[y0:y1, x0:x1]
    outer = image[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
    ring_pixels = outer.size - inner.size
    if ring_pixels <= 0:
        return None
    ring_mean = (float(outer.sum()) - float(inner.sum())) / ring_pixels
    return float(inner.mean()) - ring_mean


def contrast_by_offset(clip: Clip, offsets: Iterable[int] = range(-3, 4)) -> dict[int, float]:
    """Mean box contrast when each box is compared with frame ``frame_index + offset``.

    Only frames inside the clip's window are used. Offsets with no usable boxes map to NaN.
    """
    if clip.frame_dir is None:
        raise ValueError(f"clip {clip.name} has no frames")
    images: dict[int, NDArray[np.float32]] = {}
    for i in range(clip.frame_start, clip.frame_stop):
        image = cv2.imread(str(frame_path(clip.frame_dir, i)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"cannot decode frame {i} of {clip.name}")
        images[i] = image.astype(np.float32)

    ann = clip.annotations
    result: dict[int, float] = {}
    for offset in offsets:
        values = []
        for frame, box in zip(ann.frame_index.tolist(), ann.boxes, strict=True):
            image = images.get(frame + offset)
            if image is None:
                continue
            value = box_contrast(image, box)
            if value is not None:
                values.append(value)
        result[offset] = float(np.mean(values)) if values else float("nan")
    return result
