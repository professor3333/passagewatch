"""Letterbox preprocessing for YOLOX, and the box transforms between frame and network input.

Training and serving must use exactly this code (``CLAUDE.md``: train/serve parity):

1. The grayscale frame is resized by one scale ``r = min(H / h, W / w)``, preserving its
   aspect ratio, with bilinear interpolation to ``(int(w * r), int(h * r))`` pixels (YOLOX's
   rounding).
2. It is placed at the **top-left** of an ``H x W`` canvas filled with gray ``PAD_VALUE``.
3. The single channel is repeated to 3 channels, and values stay in ``[0, 255]`` as float32,
   without mean/std normalization: this is the input the COCO-pretrained YOLOX expects.
   A 3-channel image (``letterbox-temporal3-v1``, :mod:`passagewatch.preprocessing.temporal`)
   is resized and padded the same way, channel by channel, and kept as it is.

Boxes in the internal convention (``docs/counting_policy.md`` §1) map to network-input
pixels by multiplying by ``r`` and back by dividing by ``r``; the padding is at the right and
bottom, so there is no offset.

``PREPROCESSING_VERSION`` names the single-frame behavior. Any change to it needs a new
version, and models record the version they were trained with.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

PREPROCESSING_VERSION = "letterbox-gray3-v1"
PAD_VALUE = 114
STRIDE = 32


@dataclass(frozen=True)
class InputSize:
    """Network input size; both sides must be multiples of the model's largest stride."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"invalid input size {self.height}x{self.width}")
        if self.height % STRIDE or self.width % STRIDE:
            raise ValueError(f"input size {self.height}x{self.width} must be multiples of {STRIDE}")


@dataclass(frozen=True)
class Letterbox:
    """How one frame was placed in the network input."""

    scale: float
    frame_height: int
    frame_width: int
    resized_height: int
    resized_width: int

    def boxes_to_input(self, boxes: NDArray[np.float64]) -> NDArray[np.float64]:
        return boxes * self.scale

    def boxes_to_frame(self, boxes: NDArray[np.float64]) -> NDArray[np.float64]:
        return boxes / self.scale


def letterbox_for(frame_height: int, frame_width: int, size: InputSize) -> Letterbox:
    scale = min(size.height / frame_height, size.width / frame_width)
    return Letterbox(
        scale=scale,
        frame_height=frame_height,
        frame_width=frame_width,
        resized_height=int(frame_height * scale),
        resized_width=int(frame_width * scale),
    )


def letterbox_image(
    image: NDArray[np.uint8], size: InputSize
) -> tuple[NDArray[np.uint8], Letterbox]:
    """Resize and pad a grayscale ``h x w`` or 3-channel ``h x w x 3`` uint8 image; returns
    the ``H x W`` (or ``H x W x 3``) canvas, padded with ``PAD_VALUE`` in every channel."""
    if image.dtype != np.uint8 or not (
        image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)
    ):
        raise ValueError(
            f"expected an h x w or h x w x 3 uint8 image, got {image.dtype} {image.shape}"
        )
    box = letterbox_for(image.shape[0], image.shape[1], size)
    target = (box.resized_width, box.resized_height)
    if image.ndim == 2:
        resized = cv2.resize(image, target, interpolation=cv2.INTER_LINEAR)
    else:
        # Channel by channel: OpenCV's 3-channel path rounds differently (by up to 3 levels),
        # and this way each channel is exactly what a single-channel frame would give.
        resized = np.stack(
            [
                cv2.resize(
                    np.ascontiguousarray(image[..., c]), target, interpolation=cv2.INTER_LINEAR
                )
                for c in range(3)
            ],
            axis=-1,
        )
    canvas = np.full((size.height, size.width, *image.shape[2:]), PAD_VALUE, dtype=np.uint8)
    canvas[: box.resized_height, : box.resized_width] = resized
    return canvas, box


def letterbox_gray(gray: NDArray[np.uint8], size: InputSize) -> tuple[NDArray[np.uint8], Letterbox]:
    """Resize and pad one grayscale frame; returns the ``H x W`` uint8 canvas."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"expected a 2-D uint8 frame, got {gray.dtype} {gray.shape}")
    return letterbox_image(gray, size)


def to_network_input(canvases: NDArray[np.uint8]) -> torch.Tensor:
    """Canvases -> float32 ``(n, 3, H, W)`` in [0, 255]. Grayscale ``(H, W)`` or
    ``(n, H, W)`` canvases are repeated to 3 channels; 3-channel ``(H, W, 3)`` or
    ``(n, H, W, 3)`` canvases keep their channels, in order."""
    if canvases.ndim == 2 or (canvases.ndim == 3 and canvases.shape[-1] == 3):
        canvases = canvases[None]
    tensor = torch.from_numpy(np.ascontiguousarray(canvases)).to(torch.float32)
    if tensor.ndim == 3:
        return tensor[:, None].expand(-1, 3, -1, -1).contiguous()
    return tensor.permute(0, 3, 1, 2).contiguous()


def preprocess_frame(gray: NDArray[np.uint8], size: InputSize) -> tuple[torch.Tensor, Letterbox]:
    """The full serving-time preprocessing of one frame: ``(1, 3, H, W)`` and its letterbox."""
    canvas, box = letterbox_gray(gray, size)
    return to_network_input(canvas), box
