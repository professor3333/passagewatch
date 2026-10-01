"""Training samples for YOLOX: frames from manifest clips, with boxes as YOLOX targets.

Every ``frame_stride``-th frame of each clip's window is a sample, **including frames
without fish**, which teach the detector what background looks like. Adjacent frames are
nearly identical, so a stride loses little and shortens each epoch.

A sample is built in this order:

1. read the frame (grayscale, original size) and its boxes (internal convention), with
   boxes clipped to the frame and boxes smaller than ``min_box_px`` after clipping dropped;
2. augment (training only): horizontal flip, contrast/brightness, Gaussian noise, slight
   blur. All geometry is applied to the frame and its boxes together;
3. preprocess with :func:`passagewatch.preprocessing.letterbox.letterbox_gray` and
   :func:`to_network_input`, the same functions serving uses. With augmentation off, a
   sample's image is identical to serving's input for that frame (tested).

Targets follow YOLOX: ``(max_labels, 5)`` rows ``class, cx, cy, w, h`` in input pixels,
zero-padded.

Direction: detection labels carry no direction of travel, so a flipped sample has no
left/right label to swap. Direction is derived later from trajectories; any future
direction-labelled training data must swap left and right when flipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import Dataset

from passagewatch.ingestion.cfc import CfcLayout, Clip, frame_path, load_clip
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.preprocessing.letterbox import InputSize, letterbox_gray, to_network_input


class AugmentConfig(BaseModel):
    """Modest augmentation only: no color jitter, rotations, or crops that lose small fish."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    flip_probability: float = Field(default=0.5, ge=0, le=1)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    brightness_range: tuple[float, float] = (-15.0, 15.0)
    noise_probability: float = Field(default=0.3, ge=0, le=1)
    noise_sigma: float = Field(default=6.0, ge=0)
    blur_probability: float = Field(default=0.2, ge=0, le=1)
    blur_sigma: float = Field(default=0.8, ge=0)


@dataclass(frozen=True)
class SampleRef:
    clip_index: int
    frame_index: int


def clip_boxes(
    boxes: NDArray[np.float64], width: int, height: int, min_px: float
) -> NDArray[np.float64]:
    """Clip boxes to the frame and drop those smaller than ``min_px`` on either side."""
    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)
    keep = (clipped[:, 2] - clipped[:, 0] >= min_px) & (clipped[:, 3] - clipped[:, 1] >= min_px)
    return clipped[keep]


def flip_horizontal(
    gray: NDArray[np.uint8], boxes: NDArray[np.float64]
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    width = gray.shape[1]
    flipped = boxes.copy()
    flipped[:, 0] = width - boxes[:, 2]
    flipped[:, 2] = width - boxes[:, 0]
    return np.ascontiguousarray(gray[:, ::-1]), flipped


def photometric(
    gray: NDArray[np.uint8], config: AugmentConfig, rng: np.random.Generator
) -> NDArray[np.uint8]:
    image = gray.astype(np.float32)
    image = image * rng.uniform(*config.contrast_range) + rng.uniform(*config.brightness_range)
    if rng.random() < config.noise_probability:
        image += rng.normal(0.0, config.noise_sigma, image.shape).astype(np.float32)
    if rng.random() < config.blur_probability and config.blur_sigma > 0:
        image = np.asarray(cv2.GaussianBlur(image, (0, 0), config.blur_sigma), dtype=np.float32)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def to_targets(boxes: NDArray[np.float64], max_labels: int) -> torch.Tensor:
    """YOLOX targets: class 0, center x/y, width, height (input pixels), zero-padded."""
    targets = torch.zeros(max_labels, 5, dtype=torch.float32)
    boxes = boxes[:max_labels]
    if len(boxes):
        targets[: len(boxes), 1] = torch.from_numpy((boxes[:, 0] + boxes[:, 2]) / 2)
        targets[: len(boxes), 2] = torch.from_numpy((boxes[:, 1] + boxes[:, 3]) / 2)
        targets[: len(boxes), 3] = torch.from_numpy(boxes[:, 2] - boxes[:, 0])
        targets[: len(boxes), 4] = torch.from_numpy(boxes[:, 3] - boxes[:, 1])
    return targets


class FrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        layout: CfcLayout,
        clips: Sequence[tuple[str, ClipMetadata]],
        input_size: InputSize,
        *,
        frame_stride: int = 1,
        augment: AugmentConfig | None = None,
        max_labels: int = 64,
        min_box_px: float = 2.0,
        seed: int = 0,
    ) -> None:
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self.input_size = input_size
        self.augment = augment if augment is not None and augment.enabled else None
        self.max_labels = max_labels
        self.min_box_px = min_box_px
        self.seed = seed
        self.epoch = 0
        self.clips: list[Clip] = [load_clip(layout, loc, meta) for loc, meta in clips]
        self.samples = [
            SampleRef(ci, f)
            for ci, clip in enumerate(self.clips)
            for f in range(clip.frame_start, clip.frame_stop, frame_stride)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Augmentation randomness depends on (seed, epoch, index): reproducible on resume."""
        self.epoch = epoch

    def raw(self, index: int) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
        """The original frame and its clipped boxes, before augmentation and letterboxing."""
        ref = self.samples[index]
        clip = self.clips[ref.clip_index]
        assert clip.frame_dir is not None
        gray = cv2.imread(str(frame_path(clip.frame_dir, ref.frame_index)), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot decode frame {ref.frame_index} of {clip.name}")
        boxes = clip.annotations.boxes[clip.annotations.frame_index == ref.frame_index]
        h, w = gray.shape
        return np.asarray(gray, dtype=np.uint8), clip_boxes(boxes, w, h, self.min_box_px)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        gray, boxes = self.raw(index)
        if self.augment is not None:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            if rng.random() < self.augment.flip_probability:
                gray, boxes = flip_horizontal(gray, boxes)
            gray = photometric(gray, self.augment, rng)
        canvas, letterbox = letterbox_gray(gray, self.input_size)
        image = to_network_input(canvas)[0]
        return image, to_targets(letterbox.boxes_to_input(boxes), self.max_labels)
