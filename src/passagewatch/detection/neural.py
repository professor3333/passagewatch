"""YOLOX detectors: construction, COCO-pretrained initialization, and output decoding.

The model code is vendored in ``passagewatch.detection.yolox`` (YOLOX commit 6ddff48).
Models are built for one class ("fish"). From the COCO weights, everything is loaded except
the class-prediction layers, whose shape depends on the number of classes; loading refuses
to skip anything else, so a silent partial initialization cannot happen.

Inputs are float tensors ``(n, 3, H, W)`` with values in ``[0, 255]`` and no mean/std
normalization, as in YOLOX (``legacy=False``); see ``passagewatch.preprocessing``.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchvision
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from passagewatch.detection.yolox.yolo_head import YOLOXHead
from passagewatch.detection.yolox.yolo_pafpn import YOLOPAFPN
from passagewatch.detection.yolox.yolox import YOLOX
from passagewatch.ingestion.download import USER_AGENT

ModelSize = Literal["tiny", "s"]

# (depth multiplier, width multiplier), as in YOLOX's exps/default.
SIZES: dict[str, tuple[float, float]] = {"tiny": (0.33, 0.375), "s": (0.33, 0.50)}
IN_CHANNELS = [256, 512, 1024]
# Parameters whose shape depends on the number of classes; never loaded from COCO.
CLASS_DEPENDENT_PREFIX = "head.cls_preds."


def build_yolox(size: ModelSize, num_classes: int = 1) -> YOLOX:
    """A YOLOX model with YOLOX's default initialization (BN eps/momentum, head biases)."""
    depth, width = SIZES[size]
    backbone = YOLOPAFPN(depth, width, in_channels=IN_CHANNELS, depthwise=False)
    head = YOLOXHead(num_classes, width, in_channels=IN_CHANNELS, depthwise=False)
    model = YOLOX(backbone, head)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    head.initialize_biases(1e-2)
    return model


class PretrainedWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_pretrained_registry(path: Path) -> dict[str, PretrainedWeights]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {name: PretrainedWeights.model_validate(w) for name, w in data["weights"].items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pretrained(weights: PretrainedWeights, cache_dir: Path) -> Path:
    """Download ``weights`` into ``cache_dir`` once, verifying size and SHA-256."""
    path = cache_dir / Path(weights.url).name
    if path.is_file() and path.stat().st_size == weights.size and _sha256(path) == weights.sha256:
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(weights.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    actual = _sha256(part)
    if part.stat().st_size != weights.size or actual != weights.sha256:
        part.unlink()
        raise ValueError(f"{weights.url}: checksum mismatch (got {actual})")
    part.replace(path)
    return path


@dataclass(frozen=True)
class InitReport:
    loaded: int
    skipped: tuple[str, ...]


def load_coco_weights(model: nn.Module, checkpoint: Path) -> InitReport:
    """Load COCO weights into ``model``, skipping only the class-prediction layers."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source: dict[str, torch.Tensor] = state.get("model", state)
    own = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in own and own[k].shape == v.shape}
    skipped = tuple(sorted(k for k in own if k not in compatible))
    unexpected = [k for k in skipped if not k.startswith(CLASS_DEPENDENT_PREFIX)]
    if unexpected:
        raise ValueError(f"pretrained weights do not cover {unexpected}")
    model.load_state_dict(compatible, strict=False)
    return InitReport(loaded=len(compatible), skipped=skipped)


@dataclass(frozen=True)
class ImageDetections:
    """Detections of one image in network-input pixels: ``(n, 4)`` xyxy boxes and scores."""

    boxes: torch.Tensor
    scores: torch.Tensor


def decode_detections(
    outputs: torch.Tensor, score_threshold: float, nms_iou: float
) -> list[ImageDetections]:
    """Turn eval-mode YOLOX outputs ``(n, anchors, 5 + 1)`` into per-image detections.

    The score is objectness x class probability, as in YOLOX's postprocess.
    """
    results = []
    for image in outputs:
        scores = image[:, 4] * image[:, 5]
        keep = scores >= score_threshold
        cxcywh, kept_scores = image[keep, :4], scores[keep]
        boxes = torch.cat([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], 1)
        selected = torchvision.ops.nms(boxes, kept_scores, nms_iou)
        results.append(ImageDetections(boxes=boxes[selected], scores=kept_scores[selected]))
    return results


def select_device(preferred: str = "auto") -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU. Nothing requires CUDA."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
