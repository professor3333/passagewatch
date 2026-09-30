"""A tiny synthetic CFC tree on disk, in the publisher's layout."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

LOCATION = "kenai-train"
WIDTH, HEIGHT = 40, 60
NUM_FRAMES = 30


def mot_line(frame: int, track: int, left: float, top: float, w: float, h: float) -> str:
    return f"{frame},{track},{left:.3f},{top:.3f},{w:.3f},{h:.3f},1.0,-1,-1,-1\n"


@dataclass
class FakeCfc:
    root: Path

    @property
    def annotations(self) -> Path:
        return self.root / "fish_counting_annotations/annotations"

    @property
    def tiny(self) -> Path:
        return self.root / "tiny_dataset/tiny_dataset"

    def add_clip(
        self,
        name: str,
        rows: Iterable[str],
        *,
        window: Iterable[int] | None = range(10, 20),
        tiny_rows: Iterable[str] | None = None,
        image_size: tuple[int, int] = (WIDTH, HEIGHT),
        location: str = LOCATION,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        rows = list(rows)
        clip_dir = self.annotations / location / name
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "gt.txt").write_text("".join(rows))

        tiny_dir = self.tiny / "annotations-tiny" / location / name
        tiny_dir.mkdir(parents=True, exist_ok=True)
        (tiny_dir / "gt_tiny.txt").write_text("".join(rows if tiny_rows is None else tiny_rows))

        if window is not None:
            frames = self.tiny / "raw" / location / name
            frames.mkdir(parents=True, exist_ok=True)
            image = np.full((image_size[1], image_size[0]), 128, dtype=np.uint8)
            for i in window:
                cv2.imwrite(str(frames / f"{i}.jpg"), image)

        entry: dict[str, Any] = {
            "clip_name": name,
            "num_frames": NUM_FRAMES,
            "framerate": 10.0,
            "width": WIDTH,
            "height": HEIGHT,
            "x_meter_start": -1.0,
            "x_meter_stop": 1.0,
            "y_meter_start": 5.0,
            "y_meter_stop": 0.5,
        }
        entry.update(metadata or {})
        self.add_metadata(entry, location)

    def add_metadata(self, entry: dict[str, Any], location: str = LOCATION) -> None:
        for path in (
            self.root / f"fish_counting_metadata/metadata/{location}.json",
            self.tiny / f"metadata-tiny/{location}.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            entries = json.loads(path.read_text()) if path.exists() else []
            entries.append(entry)
            path.write_text(json.dumps(entries))


@pytest.fixture
def fake_cfc(tmp_path: Path) -> FakeCfc:
    return FakeCfc(tmp_path / "cfc")
