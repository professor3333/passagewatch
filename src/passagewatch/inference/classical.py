"""The classical pipeline for one clip: frames -> detections -> trajectories."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from passagewatch.detection.classical import (
    DetectorConfig,
    FrameDetections,
    boxes_in_meters,
    clip_scale,
    detect_clip,
    load_analysis_frames,
)
from passagewatch.ingestion.cfc import Clip, frame_path
from passagewatch.ingestion.mot import BoxAnnotations
from passagewatch.tracking.kalman import (
    KalmanTracker,
    TrackerConfig,
    Trajectory,
    trajectories_to_annotations,
)


class ClassicalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()

    def sha256(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_classical_config(path: Path) -> ClassicalConfig:
    with path.open("r", encoding="utf-8") as fh:
        return ClassicalConfig.model_validate(yaml.safe_load(fh))


@dataclass(frozen=True)
class ClipRun:
    tracks: BoxAnnotations
    trajectories: list[Trajectory]
    frame_detections: list[FrameDetections]
    seconds: dict[str, float]

    @property
    def detections(self) -> int:
        return sum(len(d.scores) for d in self.frame_detections)


def run_clip(clip: Clip, config: ClassicalConfig) -> ClipRun:
    """Run the classical pipeline over the clip's frame window."""
    if clip.frame_dir is None:
        raise ValueError(f"clip {clip.name} has no frames")
    started = time.perf_counter()
    paths = [str(frame_path(clip.frame_dir, i)) for i in range(clip.frame_start, clip.frame_stop)]
    frames, original_shape = load_analysis_frames(paths, config.detector.max_analysis_height)
    decoded = time.perf_counter()

    scale = clip_scale(clip.metadata, original_shape, frames.shape[1:])
    detections = detect_clip(frames, scale, config.detector)
    detected = time.perf_counter()

    centers = [boxes_in_meters(d.boxes, scale) for d in detections]
    trajectories = KalmanTracker(config.tracker).run(
        detections, centers, frame_offset=clip.frame_start
    )
    tracked = time.perf_counter()
    return ClipRun(
        tracks=trajectories_to_annotations(trajectories),
        trajectories=trajectories,
        frame_detections=detections,
        seconds={
            "decode": decoded - started,
            "detect": detected - decoded,
            "track": tracked - detected,
        },
    )
