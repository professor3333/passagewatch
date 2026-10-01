from __future__ import annotations

import numpy as np
import pytest
import torch

from passagewatch.preprocessing.letterbox import (
    PAD_VALUE,
    InputSize,
    letterbox_for,
    letterbox_gray,
    preprocess_frame,
    to_network_input,
)

SIZE = InputSize(height=896, width=416)


def test_input_size_must_be_multiples_of_32() -> None:
    with pytest.raises(ValueError, match="multiples of 32"):
        InputSize(height=900, width=416)


@pytest.mark.parametrize(
    ("frame", "scale"),
    [
        ((1933, 789), 896 / 1933),  # tall far-range frame: height limits
        ((624, 288), 896 / 624),  # small near frame: upscaled, height still limits
        ((835, 626), 416 / 626),  # LeftNear frame, closer to square: width limits
    ],
)
def test_one_scale_preserves_the_aspect_ratio(frame: tuple[int, int], scale: float) -> None:
    box = letterbox_for(*frame, SIZE)

    assert box.scale == pytest.approx(scale)
    assert box.resized_height <= SIZE.height and box.resized_width <= SIZE.width
    assert (box.resized_height, box.resized_width) == (int(frame[0] * scale), int(frame[1] * scale))


def test_frame_is_placed_top_left_and_padded_with_gray() -> None:
    gray = np.full((1933, 789), 7, dtype=np.uint8)

    canvas, box = letterbox_gray(gray, SIZE)

    assert canvas.shape == (896, 416)
    assert (canvas[: box.resized_height, : box.resized_width] == 7).all()
    assert (canvas[:, box.resized_width :] == PAD_VALUE).all()
    assert (canvas[box.resized_height :, :] == PAD_VALUE).all()


def test_content_lands_where_its_boxes_map() -> None:
    gray = np.zeros((600, 300), dtype=np.uint8)
    gray[200:260, 100:180] = 255  # internal box (100, 200, 180, 260)

    canvas, box = letterbox_gray(gray, SIZE)
    mapped = box.boxes_to_input(np.array([[100.0, 200.0, 180.0, 260.0]]))[0]

    x0, y0, x1, y1 = (round(float(v)) for v in mapped)
    assert canvas[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2].min() == 255
    assert canvas[y0 - 4, x0:x1].max() == 0 and canvas[y0:y1, x1 + 4].max() == 0


def test_boxes_round_trip() -> None:
    box = letterbox_for(1933, 789, SIZE)
    boxes = np.array([[0.0, 0.0, 789.0, 1933.0], [12.5, 1700.25, 80.0, 1720.0]])

    np.testing.assert_allclose(
        box.boxes_to_frame(box.boxes_to_input(boxes)), boxes, rtol=0, atol=1e-9
    )


def test_network_input_repeats_gray_without_normalization() -> None:
    gray = np.arange(600 * 300, dtype=np.int64).reshape(600, 300).astype(np.uint8)

    tensor, _ = preprocess_frame(gray, SIZE)

    assert tensor.shape == (1, 3, 896, 416) and tensor.dtype == torch.float32
    assert torch.equal(tensor[0, 0], tensor[0, 1]) and torch.equal(tensor[0, 0], tensor[0, 2])
    assert tensor.min() >= 0 and tensor.max() <= 255 and tensor.max() > 1


def test_batches_and_single_frames_give_identical_tensors() -> None:
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (600, 300), dtype=np.uint8) for _ in range(3)]

    singles = [preprocess_frame(f, SIZE)[0] for f in frames]
    batch = to_network_input(np.stack([letterbox_gray(f, SIZE)[0] for f in frames]))

    assert torch.equal(torch.cat(singles), batch)
