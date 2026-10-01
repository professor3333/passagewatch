from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from passagewatch.detection.neural import (
    PretrainedWeights,
    build_yolox,
    decode_detections,
    fetch_pretrained,
    load_coco_weights,
    load_pretrained_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def targets(batch: int) -> torch.Tensor:
    """YOLOX training targets ``(n, max_labels, 5)``: class, cx, cy, w, h in input pixels."""
    t = torch.zeros(batch, 4, 5)
    t[:, 0] = torch.tensor([0.0, 64.0, 48.0, 30.0, 10.0])
    return t


def train_step(model: torch.nn.Module, device: torch.device) -> float:
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    out = model(torch.rand(2, 3, 96, 128, device=device) * 255, targets(2).to(device))
    loss = out["total_loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss)


@pytest.mark.parametrize("size", ["tiny", "s"])
def test_models_build_and_train_on_cpu(size: str) -> None:
    model = build_yolox(size)  # type: ignore[arg-type]

    loss = train_step(model, torch.device("cpu"))

    assert torch.isfinite(torch.tensor(loss))
    assert model.head.num_classes == 1


def test_eval_outputs_decode_to_boxes() -> None:
    model = build_yolox("tiny").eval()

    with torch.no_grad():
        outputs = model(torch.rand(1, 3, 96, 128) * 255)

    # 3 strides (8, 16, 32): 12*16 + 6*8 + 3*4 anchors, each cx, cy, w, h, obj, cls.
    assert outputs.shape == (1, 12 * 16 + 6 * 8 + 3 * 4, 6)
    detections = decode_detections(outputs, score_threshold=0.0, nms_iou=0.5)
    assert detections[0].boxes.shape[1] == 4
    assert len(detections[0].boxes) == len(detections[0].scores) > 0


def test_decoding_applies_score_threshold_and_nms() -> None:
    # Two overlapping boxes and one separate box; scores are obj x cls.
    outputs = torch.tensor(
        [
            [
                [50.0, 50.0, 20.0, 20.0, 0.9, 1.0],
                [51.0, 50.0, 20.0, 20.0, 0.8, 1.0],
                [150.0, 50.0, 20.0, 20.0, 0.9, 0.5],
                [250.0, 50.0, 20.0, 20.0, 0.1, 1.0],
            ]
        ]
    )

    result = decode_detections(outputs, score_threshold=0.3, nms_iou=0.5)[0]

    assert result.scores.tolist() == pytest.approx([0.9, 0.45])
    assert result.boxes[0].tolist() == pytest.approx([40.0, 40.0, 60.0, 60.0])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS not available")
def test_model_moves_between_devices() -> None:
    # The vendored head caches coordinate grids; they must follow the device.
    model = build_yolox("tiny")
    for device in ("cpu", "mps", "cpu"):
        assert torch.isfinite(torch.tensor(train_step(model, torch.device(device))))


def test_coco_weights_load_everything_but_the_class_layers(tmp_path: Path) -> None:
    coco = build_yolox("tiny", num_classes=80)
    torch.save({"model": coco.state_dict()}, tmp_path / "coco.pth")
    model = build_yolox("tiny", num_classes=1)

    report = load_coco_weights(model, tmp_path / "coco.pth")

    assert report.skipped and all(k.startswith("head.cls_preds.") for k in report.skipped)
    assert report.loaded == len(model.state_dict()) - len(report.skipped)
    key = "backbone.backbone.stem.conv.conv.weight"
    assert torch.equal(model.state_dict()[key], coco.state_dict()[key])


def test_incomplete_weights_are_refused(tmp_path: Path) -> None:
    state = build_yolox("tiny", num_classes=80).state_dict()
    del state["backbone.backbone.stem.conv.conv.weight"]
    torch.save({"model": state}, tmp_path / "partial.pth")

    with pytest.raises(ValueError, match="do not cover"):
        load_coco_weights(build_yolox("tiny"), tmp_path / "partial.pth")


def test_pretrained_download_is_verified(tmp_path: Path) -> None:
    source = tmp_path / "src" / "weights.pth"
    source.parent.mkdir()
    source.write_bytes(b"weights" * 100)
    good = PretrainedWeights(
        url=source.as_uri(),
        size=source.stat().st_size,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )

    path = fetch_pretrained(good, tmp_path / "cache")
    assert path.read_bytes() == source.read_bytes()

    bad = good.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_pretrained(bad, tmp_path / "cache2")
    assert not any((tmp_path / "cache2").glob("*.part"))


def test_repository_registry_lists_tiny_and_s() -> None:
    registry = load_pretrained_registry(REPO_ROOT / "configs/training/pretrained.yaml")

    assert set(registry) == {"yolox-tiny", "yolox-s"}
    assert all(
        w.url.startswith("https://github.com/Megvii-BaseDetection/") for w in registry.values()
    )


@pytest.mark.slow
@pytest.mark.parametrize(("name", "size"), [("yolox-tiny", "tiny"), ("yolox-s", "s")])
def test_real_coco_weights_initialize_the_one_class_model(name: str, size: str) -> None:
    """Downloads the published weights once (cached in models/pretrained/)."""
    registry = load_pretrained_registry(REPO_ROOT / "configs/training/pretrained.yaml")
    path = fetch_pretrained(registry[name], REPO_ROOT / "models/pretrained")

    report = load_coco_weights(build_yolox(size), path)  # type: ignore[arg-type]

    assert len(report.skipped) == 6  # weight and bias of the class layer at 3 strides
    assert report.loaded == 456
