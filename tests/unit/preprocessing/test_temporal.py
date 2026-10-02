from __future__ import annotations

import numpy as np
import pytest
import torch

from passagewatch.preprocessing.letterbox import (
    PAD_VALUE,
    InputSize,
    letterbox_gray,
    letterbox_image,
    to_network_input,
)
from passagewatch.preprocessing.temporal import (
    GRAY3,
    TEMPORAL3,
    blurred,
    clip_background,
    encode_frames,
    temporal_image,
)

H, W = 40, 64


def clip(n: int = 5) -> list[np.ndarray]:
    """A dark, slightly noisy clip with a bright 6 x 10 px fish moving right 4 px a frame."""
    rng = np.random.default_rng(0)
    frames = []
    for t in range(n):
        frame = np.clip(rng.normal(40, 3, (H, W)), 0, 255).astype(np.uint8)
        frame[15:21, 8 + 4 * t : 18 + 4 * t] = 200
        frames.append(frame)
    return frames


def test_background_is_the_mean_blurred_frame() -> None:
    frames = clip()

    expected = np.mean([blurred(f) for f in frames], axis=0)

    np.testing.assert_allclose(clip_background(iter(frames)), expected, rtol=1e-5, atol=1e-4)


def test_channels_are_frame_background_difference_and_motion() -> None:
    frames = clip()
    background = clip_background(frames)

    image = temporal_image(frames[1], frames[2], background)

    assert image.shape == (H, W, 3) and image.dtype == np.uint8
    np.testing.assert_array_equal(image[..., 0], frames[1])
    # Empty water is near the background (128); the fish is far above it.
    assert abs(int(image[2, 50, 1]) - 128) <= 3
    assert image[18, 15, 1] > 200
    # Motion is near zero in still water and large at the fish's leading edge.
    assert image[2, 50, 2] <= 3
    assert image[18, 18, 2] <= 3  # inside the fish in both frames, away from its edges
    assert image[18, 24, 2] > 60  # covered only in the next frame


def test_encoding_gives_one_image_per_frame_and_the_last_looks_back() -> None:
    frames = clip()
    background = clip_background(frames)

    images = list(encode_frames(TEMPORAL3, lambda: iter(frames)))

    assert len(images) == len(frames)
    for t in range(len(frames) - 1):
        np.testing.assert_array_equal(
            images[t], temporal_image(frames[t], frames[t + 1], background)
        )
    np.testing.assert_array_equal(images[-1], temporal_image(frames[-1], frames[-2], background))


def test_a_single_frame_clip_has_no_motion() -> None:
    frames = clip(1)

    (image,) = encode_frames(TEMPORAL3, lambda: iter(frames))

    assert int(image[..., 2].max()) == 0


def test_single_frame_encoding_passes_frames_through() -> None:
    frames = clip()

    images = list(encode_frames(GRAY3, lambda: iter(frames)))

    assert all(a is b for a, b in zip(images, frames, strict=True))


def test_unknown_versions_and_empty_clips_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown preprocessing"):
        list(encode_frames("letterbox-rgb-v9", lambda: iter(clip())))
    with pytest.raises(ValueError, match="at least one frame"):
        list(encode_frames(TEMPORAL3, lambda: iter([])))


def test_three_channel_images_are_letterboxed_channel_by_channel() -> None:
    frames = clip()
    image = temporal_image(frames[0], frames[1], clip_background(frames))
    size = InputSize(64, 96)

    canvas, box = letterbox_image(image, size)
    tensor = to_network_input(canvas)

    assert canvas.shape == (64, 96, 3)
    for c in range(3):
        gray_canvas, gray_box = letterbox_gray(np.ascontiguousarray(image[..., c]), size)
        assert gray_box == box
        np.testing.assert_array_equal(canvas[..., c], gray_canvas)
        assert torch.equal(tensor[0, c], torch.from_numpy(gray_canvas).float())
    assert (canvas[box.resized_height :, :, :] == PAD_VALUE).all()


def test_grayscale_batches_are_still_repeated_to_three_channels() -> None:
    canvases = np.stack([letterbox_gray(f, InputSize(64, 96))[0] for f in clip(2)])

    tensor = to_network_input(canvases)

    assert tensor.shape == (2, 3, 64, 96)
    assert torch.equal(tensor[:, 0], tensor[:, 2])
