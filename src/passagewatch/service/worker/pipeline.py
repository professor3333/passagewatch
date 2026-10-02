"""The inference pipeline of one release bundle: frames -> detections -> tracks -> counts.

Loaded once per worker. Loading refuses a bundle whose checkpoint does not match its recorded
SHA-256, whose preprocessing version differs from this code's, or whose counting policy is
unknown, so the worker cannot run a model other than the one the bundle describes.

Frames are read and detected in batches of ``batch_size``; only their detections (boxes and
scores) are kept, so memory does not grow with the recording's pixel data. Tracking then runs
sequentially over all frames, and counting applies the job's counting configuration to the
completed trajectories.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from passagewatch.calibration.audit import seed_from
from passagewatch.calibration.versions import ClipReview, get_calibration, review_clip
from passagewatch.counting.policy import (
    CFC_COMPATIBLE_V1,
    TrajectoryCount,
    count_trajectories,
    tally,
)
from passagewatch.detection.classical import FrameDetections, Scale, boxes_in_meters
from passagewatch.inference.neural import detect_frames, file_sha256, load_detector
from passagewatch.preprocessing.temporal import PREPROCESSING_VERSIONS, encode_frames
from passagewatch.service.bundle import ReleaseBundle, load_bundle
from passagewatch.service.catalog import ClipRecord
from passagewatch.service.jobs import canonical_json
from passagewatch.service.media import iter_frames
from passagewatch.tracking.kalman import (
    KalmanTracker,
    TrackerConfig,
    Trajectory,
    trajectories_to_annotations,
)

Progress = Callable[[float], None]


class BundleMismatchError(RuntimeError):
    """The bundle's files or versions do not match what it declares or this code supports."""


class Detector(Protocol):
    def detect(self, frames: list[NDArray[np.uint8]]) -> list[FrameDetections]: ...


class YoloxDetector:
    def __init__(self, checkpoint: Path, score_threshold: float, device: torch.device) -> None:
        self.loaded = load_detector(checkpoint, device)
        self.score_threshold = score_threshold

    def detect(self, frames: list[NDArray[np.uint8]]) -> list[FrameDetections]:
        return detect_frames(self.loaded, frames, self.score_threshold)


@dataclass(frozen=True)
class PipelineResult:
    trajectories: list[Trajectory]
    counts: list[TrajectoryCount]
    right: int
    left: int
    frames: int
    review: ClipReview | None = None  # when the bundle declares a calibration version

    def summary(self) -> dict[str, Any]:
        return {"right": self.right, "left": self.left, "tracks": len(self.trajectories)}


class InferencePipeline:
    def __init__(self, bundle: ReleaseBundle, detector: Detector, *, batch_size: int = 8) -> None:
        if bundle.counting_policy != CFC_COMPATIBLE_V1.version:
            raise BundleMismatchError(f"unknown counting policy {bundle.counting_policy}")
        self.bundle = bundle
        self.detector = detector
        self.tracker = TrackerConfig.model_validate(bundle.tracker.config)
        try:
            self.calibration = get_calibration(bundle.calibration_version)
        except ValueError as exc:
            raise BundleMismatchError(str(exc)) from None
        self.batch_size = batch_size

    @property
    def version(self) -> str:
        return self.bundle.pipeline_version

    @classmethod
    def load(cls, bundle_dir: Path, device: str = "cpu", batch_size: int = 8) -> InferencePipeline:
        bundle = load_bundle(bundle_dir)
        checkpoint = bundle_dir / bundle.detector.checkpoint
        if file_sha256(checkpoint) != bundle.detector.checkpoint_sha256:
            raise BundleMismatchError(f"{checkpoint} does not match the bundle's SHA-256")
        if bundle.preprocessing_version not in PREPROCESSING_VERSIONS:
            raise BundleMismatchError(
                f"bundle needs preprocessing {bundle.preprocessing_version}, "
                f"this code implements only {', '.join(PREPROCESSING_VERSIONS)}"
            )
        detector = YoloxDetector(checkpoint, bundle.detector.score_threshold, torch.device(device))
        if detector.loaded.preprocessing != bundle.preprocessing_version:
            raise BundleMismatchError(
                f"{checkpoint} was trained with {detector.loaded.preprocessing}, but the bundle "
                f"declares {bundle.preprocessing_version}"
            )
        return cls(bundle, detector, batch_size=batch_size)

    def run(
        self,
        clip: ClipRecord,
        media_path: Path,
        counting: dict[str, Any],
        progress: Progress | None = None,
    ) -> PipelineResult:
        images = encode_frames(
            self.bundle.preprocessing_version, lambda: iter_frames(media_path, clip.media_kind)
        )
        detections = self._detect(clip, images, progress)
        scale = Scale(
            sx=1.0,
            sy=1.0,
            meters_per_px_x=abs(clip.x_meter_stop - clip.x_meter_start) / clip.width,
            meters_per_px_y=abs(clip.y_meter_stop - clip.y_meter_start) / clip.height,
        )
        centers = [boxes_in_meters(d.boxes, scale) for d in detections]
        raw = KalmanTracker(self.tracker).run(detections, centers)
        # Number trajectories 1..n so artifacts, counts and the API agree on track IDs.
        trajectories = [dataclasses.replace(t, track_id=i) for i, t in enumerate(raw, 1)]
        policy = dataclasses.replace(
            CFC_COMPATIBLE_V1, line_x_normalized=float(counting["line_x_normalized"])
        )
        counts = count_trajectories(
            trajectories_to_annotations(trajectories), clip.width, clip.height, policy
        )
        total = tally(counts)
        review = None
        if self.calibration is not None:
            # Seeded by the result cache key's parts, so a cached result keeps its windows.
            seed = seed_from(
                canonical_json([clip.sha256, self.bundle.config_sha256(), canonical_json(counting)])
            )
            review = review_clip(
                self.calibration,
                trajectories,
                counts,
                meters_per_px=(scale.meters_per_px_x, scale.meters_per_px_y),
                line_x_normalized=policy.line_x_normalized,
                num_frames=clip.num_frames,
                framerate=clip.framerate,
                seed=seed,
            )
        if progress is not None:
            progress(1.0)
        return PipelineResult(
            trajectories, counts, total.right, total.left, len(detections), review
        )

    def _detect(
        self,
        clip: ClipRecord,
        frames: Iterable[NDArray[np.uint8]],
        progress: Progress | None,
    ) -> list[FrameDetections]:
        detections: list[FrameDetections] = []
        batch: list[NDArray[np.uint8]] = []
        for index, frame in enumerate(frames):
            if frame.shape[:2] != (clip.height, clip.width):
                raise ValueError(
                    f"frame {index} is {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {clip.width}x{clip.height}"
                )
            batch.append(frame)
            if len(batch) == self.batch_size:
                detections.extend(self.detector.detect(batch))
                batch = []
                if progress is not None:
                    progress(0.95 * len(detections) / clip.num_frames)
        if batch:
            detections.extend(self.detector.detect(batch))
        if len(detections) != clip.num_frames:
            raise ValueError(f"read {len(detections)} frames, expected {clip.num_frames}")
        return detections
