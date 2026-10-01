from __future__ import annotations

import itertools
import json
import math
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torchvision.ops import box_iou

from passagewatch.detection.neural import build_yolox, decode_detections
from passagewatch.ingestion.cfc import CfcLayout
from passagewatch.preprocessing.letterbox import InputSize, preprocess_frame
from passagewatch.training.data import (
    AugmentConfig,
    FrameDataset,
    clip_boxes,
    flip_horizontal,
)
from passagewatch.training.yolox_train import TrainConfig, Trainer, learning_rate

from .conftest import LOCATION, FakeCfc, clip_name, mot_line

SIZE = InputSize(96, 64)
H, W = 60, 40


def add_clip(fake: FakeCfc, label: str, *, window: range = range(10, 20)) -> str:
    """Bright 8 x 4 target moving right; frames outside 12..17 have no fish."""
    name = clip_name(label)
    rows = [mot_line(f + 1, 1, 1 + 3 * (f - 10), 30, 8, 4) for f in range(12, 18)]
    fake.add_clip(name, rows, window=None)
    frames = fake.tiny / "raw" / LOCATION / name
    frames.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(zlib.crc32(label.encode()))
    for f in window:
        image = np.clip(rng.normal(60, 4, (H, W)), 0, 255).astype(np.uint8)
        if 12 <= f < 18:
            x = 3 * (f - 10)
            image[30:34, x : x + 8] = 220
        cv2.imwrite(str(frames / f"{f}.png"), image)
        cv2.imwrite(str(frames / f"{f}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        (frames / f"{f}.png").unlink()
    return name


def dataset(fake: FakeCfc, labels: list[str], **kwargs: object) -> FrameDataset:
    for label in labels:
        add_clip(fake, label)
    layout = CfcLayout.tiny(fake.root)
    clips = layout.metadata(LOCATION).clips
    return FrameDataset(layout, [(LOCATION, clips[clip_name(lb)]) for lb in labels], SIZE, **kwargs)  # type: ignore[arg-type]


# -- data -------------------------------------------------------------------------------


def test_training_samples_match_serving_preprocessing(fake_cfc: FakeCfc) -> None:
    data = dataset(fake_cfc, ["parity"], augment=None)

    for index in range(len(data)):
        gray, boxes = data.raw(index)
        image, targets = data[index]
        served, letterbox = preprocess_frame(gray, SIZE)
        assert torch.equal(image, served[0])
        expected = letterbox.boxes_to_input(boxes)
        n = len(expected)
        np.testing.assert_allclose(
            targets[:n, 1:].numpy(),
            np.stack(
                [
                    (expected[:, 0] + expected[:, 2]) / 2,
                    (expected[:, 1] + expected[:, 3]) / 2,
                    expected[:, 2] - expected[:, 0],
                    expected[:, 3] - expected[:, 1],
                ],
                axis=1,
            ),
            rtol=1e-6,
        )
        assert not targets[n:].any()


def test_stride_keeps_frames_without_fish(fake_cfc: FakeCfc) -> None:
    data = dataset(fake_cfc, ["stride"], frame_stride=3, augment=None)

    frames = [ref.frame_index for ref in data.samples]
    with_fish = [bool(data[i][1].any()) for i in range(len(data))]

    assert frames == [10, 13, 16, 19]
    assert with_fish == [False, True, True, False]


def test_boxes_are_clipped_and_slivers_dropped() -> None:
    boxes = np.array([[-5.0, 10.0, 15.0, 20.0], [38.5, 5.0, 45.0, 9.0], [10.0, 10.0, 20.0, 20.0]])

    clipped = clip_boxes(boxes, width=40, height=60, min_px=2.0)

    np.testing.assert_array_equal(clipped, [[0.0, 10.0, 15.0, 20.0], [10.0, 10.0, 20.0, 20.0]])


def test_flip_mirrors_pixels_and_boxes_together() -> None:
    gray = np.zeros((20, 40), dtype=np.uint8)
    gray[5:9, 2:10] = 255
    boxes = np.array([[2.0, 5.0, 10.0, 9.0]])

    flipped, fboxes = flip_horizontal(gray, boxes)

    np.testing.assert_array_equal(fboxes, [[30.0, 5.0, 38.0, 9.0]])
    assert flipped[5:9, 30:38].min() == 255 and flipped[:, :30].max() == 0
    again, back = flip_horizontal(flipped, fboxes)
    assert np.array_equal(again, gray) and np.array_equal(back, boxes)


def test_augmentation_is_reproducible_per_epoch(fake_cfc: FakeCfc) -> None:
    data = dataset(fake_cfc, ["aug"], augment=AugmentConfig(flip_probability=0.5), seed=3)

    data.set_epoch(0)
    first = [data[i][0] for i in range(len(data))]
    again = [data[i][0] for i in range(len(data))]
    data.set_epoch(1)
    other = [data[i][0] for i in range(len(data))]

    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))


# -- training ---------------------------------------------------------------------------


def config(**overrides: object) -> TrainConfig:
    base = {
        "name": "test",
        "pretrained": None,
        "input_height": SIZE.height,
        "input_width": SIZE.width,
        "frame_stride": 1,
        "batch_size": 4,
        "epochs": 2,
        "num_workers": 0,
        "log_every_iters": 1,
        "checkpoint_every_iters": 1000,
        "augment": {"enabled": False},
    }
    return TrainConfig.model_validate(base | overrides)


def test_learning_rate_warms_up_then_decays_to_the_floor() -> None:
    cfg = config(epochs=10, warmup_epochs=1.0, min_lr_ratio=0.1)
    peak = cfg.lr_per_image * cfg.batch_size
    lrs = [learning_rate(cfg, i, 10) for i in range(100)]

    assert lrs[0] < lrs[5] < lrs[9] == pytest.approx(peak)
    assert lrs[10] == pytest.approx(peak)
    assert lrs[99] == pytest.approx(0.1 * peak, rel=0.01)
    assert all(a >= b for a, b in itertools.pairwise(lrs[10:]))


def test_smoke_training_writes_checkpoints_and_metrics(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    data = dataset(fake_cfc, ["smoke"], augment=None)
    trainer = Trainer(config(epochs=1), data, tmp_path / "run", device="cpu")

    state = trainer.train()

    assert state.epoch == 1 and state.iteration == math.ceil(len(data) / 4)
    assert (tmp_path / "run" / "latest.pt").is_file()
    assert (tmp_path / "run" / "epoch-001.pt").is_file()
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == state.iteration and "total_loss" in json.loads(lines[0])
    info = json.loads((tmp_path / "run" / "run.json").read_text())
    assert info["preprocessing_version"] == "letterbox-gray3-v1" and info["seeds"]["torch"] == 0


def test_resume_reproduces_an_uninterrupted_run(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    data = dataset(fake_cfc, ["resume"], augment=AugmentConfig())  # 10 samples: 3 iters/epoch
    cfg = config(epochs=3, augment=AugmentConfig().model_dump())

    straight = Trainer(cfg, data, tmp_path / "a", device="cpu")
    straight.train(max_iters=7)

    interrupted = Trainer(cfg, data, tmp_path / "b", device="cpu")
    interrupted.train(max_iters=2)
    resumed = Trainer(cfg, data, tmp_path / "b", device="cpu")
    resumed.resume(tmp_path / "b" / "latest.pt")
    resumed.train(max_iters=5)  # crosses the end of epoch 1 into epoch 3

    assert resumed.state == straight.state
    a, b = straight.model.state_dict(), resumed.model.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_resume_refuses_a_different_config(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    data = dataset(fake_cfc, ["mismatch"], augment=None)
    Trainer(config(), data, tmp_path / "run", device="cpu").train(max_iters=1)

    other = Trainer(config(seed=1), data, tmp_path / "run", device="cpu")
    with pytest.raises(ValueError, match="different training config"):
        other.resume(tmp_path / "run" / "latest.pt")


def test_head_only_phase_freezes_backbone_weights_and_statistics(
    fake_cfc: FakeCfc, tmp_path: Path
) -> None:
    torch.save({"model": build_yolox("tiny", 80).state_dict()}, tmp_path / "coco.pth")
    data = dataset(fake_cfc, ["frozen"], augment=None)
    cfg = config(pretrained="fake-coco", head_only_epochs=1, epochs=2)
    trainer = Trainer(cfg, data, tmp_path / "run", device="cpu", init_weights=tmp_path / "coco.pth")
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}

    trainer.train(max_iters=2)

    after = trainer.model.state_dict()
    backbone = [k for k in before if k.startswith("backbone.")]
    head = [k for k in before if k.startswith("head.") and before[k].is_floating_point()]
    assert all(torch.equal(before[k], after[k]) for k in backbone)  # incl. BN running stats
    assert any(not torch.equal(before[k], after[k]) for k in head)


def test_pretrained_config_needs_weights(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    data = dataset(fake_cfc, ["noweights"], augment=None)

    with pytest.raises(ValueError, match="needs pretrained weights"):
        Trainer(config(pretrained="yolox-tiny"), data, tmp_path / "run", device="cpu")


def _tiny_set() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Six frames, each with one bright 24 x 12 px target (larger than the 8 px grid cell)."""
    rng = np.random.default_rng(0)
    images, targets = [], []
    for i in range(6):
        frame = np.clip(rng.normal(60, 4, (SIZE.height, SIZE.width)), 0, 255).astype(np.float32)
        x = 4 + 6 * i
        frame[40:52, x : x + 24] = 220
        images.append(torch.from_numpy(frame)[None].expand(3, -1, -1).contiguous())
        target = torch.zeros(8, 5)
        target[0] = torch.tensor([0.0, x + 12.0, 46.0, 24.0, 12.0])
        targets.append(target)
    return images, targets


def _train_tiny_set(tmp_path: Path, steps: int, seed: int) -> tuple[Trainer, list[float]]:
    images, targets = _tiny_set()

    class Frames(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
        # One batch of all six frames per epoch.
        def __init__(self) -> None:
            self.clips: list[object] = []

        def __len__(self) -> int:
            return len(images)

        def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
            return images[i], targets[i]

        def set_epoch(self, epoch: int) -> None:
            pass

    cfg = config(
        epochs=steps,
        batch_size=6,
        lr_per_image=0.01 / 6,
        warmup_epochs=0.0,
        min_lr_ratio=1.0,
        weight_decay=0.0,
        log_every_iters=10,
        save_every_epochs=steps,
        seed=seed,
    )
    trainer = Trainer(cfg, Frames(), tmp_path / "run", device="cpu")  # type: ignore[arg-type]
    # YOLOX's BatchNorm momentum (0.03) makes eval-mode statistics lag training by a few
    # hundred steps; a faster momentum lets this tiny run be judged in eval mode.
    for module in trainer.model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = 0.3
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer.train()
    finally:
        torch.set_num_threads(threads)
    metrics = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    return trainer, [json.loads(line)["total_loss"] for line in metrics]


def test_training_reduces_the_loss(tmp_path: Path) -> None:
    """Fast check that the pipeline learns at all: targets, gradients and optimizer work.

    Over 100 steps the loss fell by 19-53% for each of 8 seeds checked, so this holds on any
    hardware; a pipeline that cannot learn stays near its initial loss.
    """
    _, losses = _train_tiny_set(tmp_path, steps=100, seed=0)

    assert np.mean(losses[-3:]) < 0.9 * np.mean(losses[:3])


@pytest.mark.slow
def test_overfits_a_tiny_set(tmp_path: Path) -> None:
    """The model finds the target it was trained on, in eval mode.

    Seed-sensitive: from scratch, 300 steps found the target for 4 of 5 seeds checked, so
    this is a slow, local sanity check rather than a CI gate (where CPU differences between
    runners act like a different seed).
    """
    trainer, losses = _train_tiny_set(tmp_path, steps=300, seed=1)

    assert np.mean(losses[-5:]) < 0.6 * np.mean(losses[:3])
    images, _ = _tiny_set()
    with torch.no_grad():
        detections = decode_detections(
            trainer.model.eval()(images[2][None]), score_threshold=0.3, nms_iou=0.5
        )[0]
    assert len(detections.boxes) > 0
    assert box_iou(detections.boxes[:1], torch.tensor([[16.0, 40.0, 40.0, 52.0]])).item() > 0.5


def test_epoch_snapshots_follow_save_every_epochs(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    data = dataset(fake_cfc, ["snapshots"], augment=None)

    Trainer(config(epochs=5, save_every_epochs=2), data, tmp_path / "run", device="cpu").train()

    snapshots = sorted(p.name for p in (tmp_path / "run").glob("epoch-*.pt"))
    assert snapshots == ["epoch-002.pt", "epoch-004.pt", "epoch-005.pt"]


def test_repository_training_config_is_valid() -> None:
    from passagewatch.training.yolox_train import load_train_config

    path = Path(__file__).resolve().parents[2] / "configs/training/yolox-tiny-v1.yaml"
    cfg = load_train_config(path)

    assert (cfg.model_size, cfg.pretrained) == ("tiny", "yolox-tiny")
    assert cfg.input_size == InputSize(960, 416)
    assert cfg.head_only_epochs >= 1 and cfg.backbone_lr_factor < 1
