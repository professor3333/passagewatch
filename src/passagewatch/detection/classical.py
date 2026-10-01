"""Classical fish detection: background subtraction, connected components, size filter.

Fish return brighter echoes than the slowly changing riverbed and water. For each frame:

1. Frames are processed at an analysis scale (height capped at ``max_analysis_height``).
2. The background is a temporal median over a window of nearby frames, recomputed every
   ``background_update_every`` frames. The window is centered, because recordings are
   analyzed after upload (batch), not live.
3. Foreground is where ``frame - background`` exceeds ``threshold_sigma`` times a robust
   estimate of the noise inside the sonar fan (the image outside the fan is black).
4. Morphological opening removes speckle; closing joins the parts of one fish.
5. Connected components whose area in m² (from the clip's meter extents) is within
   ``[min_area_m2, max_area_m2]`` become detections, scored by mean contrast in noise units.

Sizes are in meters so that one configuration applies to clips rendered at different
resolutions. Boxes are returned in the internal convention, in original-frame pixels.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from passagewatch.ingestion.metadata import ClipMetadata

GrayFrames = NDArray[np.uint8]


class DetectorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_analysis_height: int = Field(default=960, ge=64)
    background_window: int = Field(default=61, ge=3)
    background_stride: int = Field(default=3, ge=1)
    background_update_every: int = Field(default=5, ge=1)
    blur_sigma_px: float = Field(default=1.0, ge=0)
    threshold_sigma: float = Field(default=3.0, gt=0)
    min_contrast: float = Field(default=8.0, ge=0)
    open_px: int = Field(default=2, ge=0)
    close_px: int = Field(default=5, ge=0)
    min_area_m2: float = Field(default=0.004, ge=0)
    max_area_m2: float = Field(default=0.6, gt=0)
    fan_threshold: int = Field(default=8, ge=0)
    # Sonar noise changes with range (image rows). With n > 1 bands, the noise level is
    # estimated separately in n horizontal bands of rows instead of once per frame.
    noise_bands: int = Field(default=1, ge=1)


@dataclass(frozen=True)
class FrameDetections:
    """Detections of one frame: boxes ``(n, 4)`` in original pixels, and scores ``(n,)``."""

    boxes: NDArray[np.float64]
    scores: NDArray[np.float64]


@dataclass(frozen=True)
class Scale:
    """Mapping between original-frame pixels and analysis pixels."""

    sx: float  # original pixels per analysis pixel, horizontally
    sy: float
    meters_per_px_x: float  # meters per original pixel
    meters_per_px_y: float

    @property
    def analysis_pixel_area_m2(self) -> float:
        return self.sx * self.meters_per_px_x * self.sy * self.meters_per_px_y


def analysis_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    if height <= max_height:
        return width, height
    factor = max_height / height
    return max(1, round(width * factor)), max_height


def clip_scale(
    meta: ClipMetadata, frame_shape: tuple[int, int], analysis_shape: tuple[int, int]
) -> Scale:
    """``frame_shape`` and ``analysis_shape`` are ``(height, width)``."""
    h, w = frame_shape
    ah, aw = analysis_shape
    return Scale(
        sx=w / aw,
        sy=h / ah,
        meters_per_px_x=abs(meta.x_meter_stop - meta.x_meter_start) / meta.width,
        meters_per_px_y=abs(meta.y_meter_stop - meta.y_meter_start) / meta.height,
    )


def _windows(n: int, config: DetectorConfig) -> list[tuple[int, int, list[int]]]:
    """``(start, stop, sample_indices)``: frames [start, stop) share one background."""
    half = config.background_window // 2
    blocks = []
    for start in range(0, n, config.background_update_every):
        stop = min(n, start + config.background_update_every)
        center = (start + stop - 1) // 2
        lo, hi = max(0, center - half), min(n, center + half + 1)
        # Shift the window inward at the clip edges so it keeps its full length.
        if hi - lo < config.background_window:
            if lo == 0:
                hi = min(n, config.background_window)
            else:
                lo = max(0, n - config.background_window)
        blocks.append((start, stop, list(range(lo, hi, config.background_stride))))
    return blocks


def _row_bands(fan: NDArray[np.bool_], count: int) -> list[tuple[int, int]]:
    """Split the rows that contain fan pixels into ``count`` bands of equal height."""
    rows = np.flatnonzero(fan.any(axis=1))
    edges = np.linspace(rows[0], rows[-1] + 1, count + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in itertools.pairwise(edges.tolist()) if b > a]


def _noise_maps(
    diff: NDArray[np.float32], fan: NDArray[np.bool_], bands: list[tuple[int, int]]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Per-row robust center (median) and spread (1.4826 x MAD), as ``(h, 1)`` columns.

    Rows above the first band and below the last use the nearest band's values.
    """
    h = diff.shape[0]
    center = np.zeros((h, 1), dtype=np.float32)
    spread = np.ones((h, 1), dtype=np.float32)
    for i, (a, b) in enumerate(bands):
        values = diff[a:b][fan[a:b]]
        if values.size == 0:
            continue
        med = float(np.median(values))
        mad = 1.4826 * float(np.median(np.abs(values - med))) or 1.0
        lo = 0 if i == 0 else a
        hi = h if i == len(bands) - 1 else b
        center[lo:hi] = med
        spread[lo:hi] = mad
    return center, spread


def _kernel(size: int) -> NDArray[np.uint8] | None:
    if size <= 1:
        return None
    return np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)), dtype=np.uint8)


def detect_clip(frames: GrayFrames, scale: Scale, config: DetectorConfig) -> list[FrameDetections]:
    """Detect in every frame of ``frames`` ``(t, h, w)``, given at analysis scale."""
    n = len(frames)
    fan = np.median(frames[:: max(1, n // 32)], axis=0) > config.fan_threshold
    fan = cv2.erode(fan.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if not fan.any():
        return [FrameDetections(np.empty((0, 4)), np.empty(0)) for _ in range(n)]
    open_k, close_k = _kernel(config.open_px), _kernel(config.close_px)
    min_area = config.min_area_m2 / scale.analysis_pixel_area_m2
    max_area = config.max_area_m2 / scale.analysis_pixel_area_m2

    bands = _row_bands(fan, config.noise_bands)
    results: list[FrameDetections] = []
    for start, stop, samples in _windows(n, config):
        background = np.median(frames[samples], axis=0).astype(np.float32)
        for t in range(start, stop):
            diff = frames[t].astype(np.float32) - background
            if config.blur_sigma_px > 0:
                diff = cv2.GaussianBlur(diff, (0, 0), config.blur_sigma_px)
            center, spread = _noise_maps(diff, fan, bands)
            threshold = np.maximum(center + config.threshold_sigma * spread, config.min_contrast)
            mask = ((diff > threshold) & fan).astype(np.uint8)
            if open_k is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
            if close_k is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            boxes: list[Sequence[float]] = []
            scores: list[float] = []
            for label in range(1, count):
                left, top, width, height, area = (int(v) for v in stats[label])
                if not min_area <= area <= max_area:
                    continue
                contrast = float(diff[labels == label].mean())
                boxes.append(
                    (
                        left * scale.sx,
                        top * scale.sy,
                        (left + width) * scale.sx,
                        (top + height) * scale.sy,
                    )
                )
                row = min(top + height // 2, len(center) - 1)
                scores.append((contrast - float(center[row, 0])) / float(spread[row, 0]))
            results.append(
                FrameDetections(
                    boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                    scores=np.asarray(scores, dtype=np.float64),
                )
            )
    return results


def load_analysis_frames(
    paths: Sequence[str], max_height: int
) -> tuple[GrayFrames, tuple[int, int]]:
    """Read grayscale frames, resized to the analysis size; also return the original shape."""
    first = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise ValueError(f"cannot decode {paths[0]}")
    h, w = first.shape
    aw, ah = analysis_size(w, h, max_height)
    out = np.empty((len(paths), ah, aw), dtype=np.uint8)
    for i, path in enumerate(paths):
        image = first if i == 0 else cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"cannot decode {path}")
        if image.shape != (h, w):
            raise ValueError(f"{path} is {image.shape[::-1]}, expected {(w, h)}")
        out[i] = (
            image
            if (aw, ah) == (w, h)
            else cv2.resize(image, (aw, ah), interpolation=cv2.INTER_AREA)
        )
    return out, (h, w)


def boxes_in_meters(boxes: NDArray[np.float64], scale: Scale) -> NDArray[np.float64]:
    """Box centers in meters ``(n, 2)``, for distance gating."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 * scale.meters_per_px_x
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 * scale.meters_per_px_y
    return np.stack([cx, cy], axis=1)
