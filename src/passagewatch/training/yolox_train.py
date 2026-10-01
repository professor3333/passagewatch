"""Fine-tune YOLOX on sonar frames, with resumable checkpoints.

Transfer learning, as planned in ``CLAUDE.md``:

1. a new one-class head on a COCO-pretrained YOLOX;
2. ``head_only_epochs`` with the pretrained backbone and neck frozen (their BatchNorm
   statistics are frozen too);
3. full fine-tuning, with the backbone and neck at ``backbone_lr_factor`` times the head's
   learning rate.

``pretrained: null`` trains from scratch (the control run): no head-only phase and one
learning rate for every layer.

Learning rate: YOLOX's linear scaling (``lr_per_image x batch_size``), a quadratic warm-up,
then cosine decay to ``min_lr_ratio`` of the peak, updated every iteration.

**Checkpoints.** ``latest.pt`` holds everything needed to continue exactly: weights,
optimizer, AMP scaler, epoch, the position within the epoch, and the Python, NumPy and
PyTorch RNG states. The sample order of every epoch is a permutation seeded by
``(seed, epoch)``, so a run resumed mid-epoch skips the batches already seen and continues
as if never interrupted (tested on CPU). After every epoch, ``epoch-NNN.pt`` keeps that
epoch's weights. **No checkpoint is called "best" here:** the released detector is chosen
afterwards by counting nMAE on kenai-val through the full pipeline, not by training loss
or AP.
"""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.utils.data import DataLoader, Sampler

from passagewatch.detection.neural import build_yolox, load_coco_weights, select_device
from passagewatch.preprocessing.letterbox import PREPROCESSING_VERSION, InputSize
from passagewatch.training.data import AugmentConfig, FrameDataset


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    model_size: Literal["tiny", "s"] = "tiny"
    pretrained: str | None = "yolox-tiny"
    input_height: int = 960
    input_width: int = 416
    frame_stride: int = Field(default=3, ge=1)
    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=30, ge=1)
    head_only_epochs: int = Field(default=1, ge=0)
    lr_per_image: float = Field(default=0.01 / 64, gt=0)
    backbone_lr_factor: float = Field(default=0.1, gt=0, le=1)
    warmup_epochs: float = Field(default=1.0, ge=0)
    min_lr_ratio: float = Field(default=0.05, ge=0, le=1)
    momentum: float = 0.9
    weight_decay: float = 5e-4
    max_labels: int = 64
    num_workers: int = Field(default=4, ge=0)
    seed: int = 0
    amp: bool = True
    log_every_iters: int = Field(default=50, ge=1)
    checkpoint_every_iters: int = Field(default=500, ge=1)
    # Keep an epoch snapshot and refresh latest.pt every N epochs (and after the last one).
    save_every_epochs: int = Field(default=1, ge=1)
    augment: AugmentConfig = AugmentConfig()

    @property
    def input_size(self) -> InputSize:
        return InputSize(self.input_height, self.input_width)


def load_train_config(path: Path) -> TrainConfig:
    with path.open("r", encoding="utf-8") as fh:
        return TrainConfig.model_validate(yaml.safe_load(fh))


class EpochSampler(Sampler[int]):
    """A seeded permutation per epoch that can start part-way through."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.start = 0

    def set_position(self, epoch: int, start: int) -> None:
        self.epoch, self.start = epoch, start

    def order(self, epoch: int) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed * 100_003 + epoch)
        return torch.randperm(self.size, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        return iter(self.order(self.epoch)[self.start :])

    def __len__(self) -> int:
        return self.size - self.start


def seed_everything(seed: int) -> None:
    random.seed(seed)
    # The global NumPy RNG is seeded for any library code that uses it; PassageWatch code
    # itself uses explicit Generators.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Backbone+neck vs head, each split into decayed weights and undecayed BN/bias."""
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        part = "backbone" if name.startswith("backbone.") else "head"
        no_decay = param.ndim <= 1  # biases and BatchNorm weights
        groups.setdefault((part, no_decay), []).append(param)
    return [
        {"params": params, "part": part, "weight_decay": 0.0 if no_decay else weight_decay}
        for (part, no_decay), params in sorted(groups.items())
    ]


def learning_rate(config: TrainConfig, iteration: int, iters_per_epoch: int) -> float:
    """Peak-relative schedule: quadratic warm-up, then cosine decay to ``min_lr_ratio``."""
    peak = config.lr_per_image * config.batch_size
    total = config.epochs * iters_per_epoch
    warmup = int(config.warmup_epochs * iters_per_epoch)
    if iteration < warmup:
        return peak * ((iteration + 1) / warmup) ** 2
    progress = (iteration - warmup) / max(1, total - warmup)
    floor = peak * config.min_lr_ratio
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    backbone: nn.Module = model.backbone  # type: ignore[assignment]
    for param in backbone.parameters():
        param.requires_grad_(trainable)
    # Frozen layers also keep their pretrained BatchNorm statistics.
    backbone.train(trainable and model.training)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_commit() -> str | None:
    return _git("rev-parse", "HEAD") or None


def git_dirty() -> bool | None:
    """True if tracked files differ from the commit (the run is not fully reproducible)."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


@dataclass
class TrainState:
    epoch: int = 0  # current epoch (0-based)
    batch_in_epoch: int = 0  # batches of the current epoch already done
    iteration: int = 0  # batches done in total


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        dataset: FrameDataset,
        out_dir: Path,
        *,
        device: str = "auto",
        init_weights: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.out_dir = out_dir
        self.device = select_device(device)
        self.metadata = metadata or {}
        seed_everything(config.seed)
        self.model = build_yolox(config.model_size)
        if init_weights is not None:
            self.init_report = load_coco_weights(self.model, init_weights)
        elif config.pretrained is not None:
            raise ValueError(f"config {config.name} needs pretrained weights {config.pretrained}")
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            param_groups(self.model, config.weight_decay),
            lr=0.0,
            momentum=config.momentum,
            nesterov=True,
        )
        self.use_amp = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.sampler = EpochSampler(len(dataset), config.seed)
        self.iters_per_epoch = math.ceil(len(dataset) / config.batch_size)
        self.state = TrainState()

    # -- checkpoints ------------------------------------------------------------------

    def _payload(self) -> dict[str, Any]:
        return {
            "format": 1,
            "config": self.config.model_dump(mode="json"),
            "preprocessing_version": PREPROCESSING_VERSION,
            "model": self.model.state_dict(),
            "state": vars(self.state).copy(),
            "metadata": self.metadata,
        }

    def save_latest(self) -> Path:
        payload = self._payload() | {
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": rng_state(),
        }
        path = self.out_dir / "latest.pt"
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def save_epoch(self, epoch: int) -> Path:
        path = self.out_dir / f"epoch-{epoch + 1:03d}.pt"
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self._payload(), tmp)
        tmp.replace(path)
        return path

    def resume(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["config"] != self.config.model_dump(mode="json"):
            raise ValueError(f"{path} was written with a different training config")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.state = TrainState(**payload["state"])
        set_rng_state(payload["rng"])

    # -- training ---------------------------------------------------------------------

    def _log(self, record: dict[str, Any]) -> None:
        with (self.out_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _write_run_info(self) -> None:
        info = {
            "config": self.config.model_dump(mode="json"),
            "preprocessing_version": PREPROCESSING_VERSION,
            "samples": len(self.dataset),
            "clips": len(self.dataset.clips),
            "iters_per_epoch": self.iters_per_epoch,
            "device": str(self.device),
            "amp": self.use_amp,
            "seeds": {
                "python": self.config.seed,
                "numpy": self.config.seed,
                "torch": self.config.seed,
                "sampler": self.config.seed,
            },
            "torch": torch.__version__,
            "python": platform.python_version(),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "metadata": self.metadata,
        }
        if hasattr(self, "init_report"):
            info["pretrained_skipped"] = list(self.init_report.skipped)
        (self.out_dir / "run.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")

    def _set_lr(self) -> float:
        lr = learning_rate(self.config, self.state.iteration, self.iters_per_epoch)
        pretrained = self.config.pretrained is not None
        for group in self.optimizer.param_groups:
            factor = (
                self.config.backbone_lr_factor
                if pretrained and group["part"] == "backbone"
                else 1.0
            )
            group["lr"] = lr * factor
        return lr

    def train(self, max_iters: int | None = None) -> TrainState:
        """Train until ``config.epochs`` are done, or ``max_iters`` more batches (for tests)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_info()
        done = 0
        while self.state.epoch < self.config.epochs:
            epoch = self.state.epoch
            head_only = self.config.pretrained is not None and epoch < self.config.head_only_epochs
            self.model.train()
            set_backbone_trainable(self.model, not head_only)
            self.dataset.set_epoch(epoch)
            self.sampler.set_position(epoch, self.state.batch_in_epoch * self.config.batch_size)
            loader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                sampler=self.sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.device.type == "cuda",
                drop_last=False,
                persistent_workers=False,
            )
            started = time.perf_counter()
            for batches_this_session, (images, targets) in enumerate(loader, 1):
                lr = self._set_lr()
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    outputs = self.model(images, targets)
                loss = outputs["total_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"loss is {float(loss)} at iteration {self.state.iteration}"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.state.iteration += 1
                self.state.batch_in_epoch += 1
                done += 1
                if self.state.iteration % self.config.log_every_iters == 0:
                    elapsed = time.perf_counter() - started
                    self._log(
                        {
                            "epoch": epoch + 1,
                            "iteration": self.state.iteration,
                            "lr": lr,
                            "head_only": head_only,
                            "images_per_second": round(
                                batches_this_session * self.config.batch_size / max(elapsed, 1e-9),
                                1,
                            ),
                            **{
                                k: float(v)
                                for k, v in outputs.items()
                                if isinstance(v, torch.Tensor | float | int)
                            },
                        }
                    )
                if self.state.iteration % self.config.checkpoint_every_iters == 0:
                    self.save_latest()
                if max_iters is not None and done >= max_iters:
                    self.save_latest()
                    return self.state
            self.state = TrainState(
                epoch=epoch + 1, batch_in_epoch=0, iteration=self.state.iteration
            )
            last = self.state.epoch == self.config.epochs
            if last or self.state.epoch % self.config.save_every_epochs == 0:
                self.save_epoch(epoch)
                self.save_latest()
        return self.state
