"""Frame encodings: how decoded grayscale frames become the detector's input images.

Two preprocessing versions exist, and a model records the one it was trained with:

- ``letterbox-gray3-v1``: the frame itself, repeated to 3 channels by
  :func:`passagewatch.preprocessing.letterbox.to_network_input`.
- ``letterbox-temporal3-v1``: three channels built from the clip around the frame,
  following the CFC authors' **Baseline++** (Kay et al., ECCV 2022; their ``convert.py``):

  1. the frame itself;
  2. the frame minus the clip's background: ``128 + (blur(frame) - background)``;
  3. motion: ``|blur(next frame) - blur(frame)|``.

  ``blur`` is a 5 x 5 Gaussian, and ``background`` is the mean of the blurred frames of the
  whole clip, both as in Baseline++. Values are clipped to ``[0, 255]``.

How this differs from Baseline++:

- Baseline++ scales channels 2 and 3 by the clip's largest absolute background difference,
  so a single bright artifact compresses the contrast of a whole clip, and the scale needs
  another full pass. Here both use a fixed scale of one gray level per level.
- Baseline++ takes the motion channel from background-subtracted frames. The background
  cancels in that difference, so it is computed from the blurred frames directly.
- Baseline++ drops the clip's last frame, which has no next frame. Every frame needs
  detections here, so the last frame uses the previous frame instead.

Temporal context comes only from the same clip. The background needs every frame of the
clip, so encoding reads the clip twice: once to average, once to encode, keeping only two
frames and the background in memory. Training and serving both use :func:`encode_frames`
(training through :func:`temporal_image` with the same inputs), which a parity test checks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from passagewatch.preprocessing.letterbox import PREPROCESSING_VERSION

GRAY3 = PREPROCESSING_VERSION
TEMPORAL3 = "letterbox-temporal3-v1"
PREPROCESSING_VERSIONS = (GRAY3, TEMPORAL3)
BLUR_KSIZE = (5, 5)
BACKGROUND_OFFSET = 128.0

Background = NDArray[np.float32]


def check_version(version: str) -> None:
    if version not in PREPROCESSING_VERSIONS:
        raise ValueError(f"unknown preprocessing version {version!r}")


def blurred(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    return np.asarray(cv2.GaussianBlur(gray.astype(np.float32), BLUR_KSIZE, 0), dtype=np.float32)


def clip_background(frames: Iterable[NDArray[np.uint8]]) -> Background:
    """The mean blurred frame of a clip, accumulated one frame at a time."""
    total: NDArray[np.float64] | None = None
    count = 0
    for frame in frames:
        b = blurred(frame).astype(np.float64)
        if total is None:
            total = b
        elif b.shape != total.shape:
            raise ValueError(f"frame {count} is {b.shape}, expected {total.shape}")
        else:
            total += b
        count += 1
    if total is None:
        raise ValueError("a clip needs at least one frame for its background")
    return (total / count).astype(np.float32)


def temporal_image(
    frame: NDArray[np.uint8], following: NDArray[np.uint8], background: Background
) -> NDArray[np.uint8]:
    """The ``h x w x 3`` uint8 image of ``frame``. ``following`` is the next frame of the
    clip, or the previous one for the last frame (or the frame itself in a 1-frame clip)."""
    current = blurred(frame)
    channels = (
        frame.astype(np.float32),
        BACKGROUND_OFFSET + (current - background),
        np.abs(blurred(following) - current),
    )
    stacked = np.stack(channels, axis=-1)
    return np.clip(np.rint(stacked), 0, 255).astype(np.uint8)


def encode_frames(
    version: str, frames: Callable[[], Iterable[NDArray[np.uint8]]]
) -> Iterator[NDArray[np.uint8]]:
    """The input image of every frame of one clip, in order.

    ``frames`` returns a fresh iterator over the clip's grayscale frames each time it is
    called; the temporal encoding calls it twice.
    """
    check_version(version)
    if version == GRAY3:
        yield from frames()
        return
    background = clip_background(frames())
    previous: NDArray[np.uint8] | None = None
    current: NDArray[np.uint8] | None = None
    for following in frames():
        if current is not None:
            yield temporal_image(current, following, background)
        previous, current = current, following
    if current is not None:
        yield temporal_image(current, previous if previous is not None else current, background)
