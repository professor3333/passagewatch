"""Neural detection for whole clips, with detections cached for cheap re-tracking.

A trained checkpoint records its preprocessing version, and frames are encoded with exactly
that version (:func:`passagewatch.preprocessing.temporal.encode_frames`), so a model is never
served with preprocessing it was not trained with. Encoded images go through the same
:func:`letterbox_image` as training; detections are mapped back to original-frame pixels
(divided by the letterbox scale) and clipped to the frame.

Detection is the expensive step. :func:`detect_clip` runs it once at a low score threshold;
the result is stored as an ``.npz`` file, and choosing a score threshold, tracking and
counting then work from the cache without the model.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from passagewatch.counting.policy import CFC_COMPATIBLE_V1, CountingPolicy
from passagewatch.detection.classical import FrameDetections, Scale, boxes_in_meters
from passagewatch.detection.neural import build_yolox, decode_detections
from passagewatch.detection.yolox.yolox import YOLOX
from passagewatch.evaluation.detection import DetectionMatch, match_detections
from passagewatch.evaluation.nmae import ClipCountError, count_clip
from passagewatch.ingestion.cfc import Clip, frame_path
from passagewatch.preprocessing.letterbox import InputSize, letterbox_image, to_network_input
from passagewatch.preprocessing.temporal import PREPROCESSING_VERSIONS, encode_frames
from passagewatch.tracking.bytetrack import ByteTrackConfig, ByteTracker
from passagewatch.tracking.kalman import (
    KalmanTracker,
    TrackerConfig,
    Trajectory,
    trajectories_to_annotations,
)

# Detections below this score are never cached; thresholds are chosen above it.
CACHE_SCORE_THRESHOLD = 0.05
NMS_IOU = 0.65  # YOLOX's evaluation default


@dataclass(frozen=True)
class LoadedDetector:
    model: YOLOX
    input_size: InputSize
    device: torch.device
    checkpoint_sha256: str
    epoch: int
    run_name: str
    preprocessing: str = PREPROCESSING_VERSIONS[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_detector(checkpoint: Path, device: torch.device) -> LoadedDetector:
    payload: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["preprocessing_version"] not in PREPROCESSING_VERSIONS:
        raise ValueError(
            f"{checkpoint} was trained with preprocessing {payload['preprocessing_version']}, "
            f"but this code implements only {', '.join(PREPROCESSING_VERSIONS)}"
        )
    config = payload["config"]
    model = build_yolox(config["model_size"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return LoadedDetector(
        model=model,
        input_size=InputSize(config["input_height"], config["input_width"]),
        device=device,
        checkpoint_sha256=file_sha256(checkpoint),
        epoch=int(payload["state"]["epoch"]),
        run_name=config["name"],
        preprocessing=payload["preprocessing_version"],
    )


@torch.no_grad()
def detect_frames(
    detector: LoadedDetector,
    frames: list[NDArray[np.uint8]],
    score_threshold: float = CACHE_SCORE_THRESHOLD,
) -> list[FrameDetections]:
    """Detect in encoded frames (grayscale, or 3-channel for temporal preprocessing); boxes
    in original-frame pixels, clipped to the frame."""
    canvases, letterboxes = zip(
        *(letterbox_image(f, detector.input_size) for f in frames), strict=True
    )
    batch = to_network_input(np.stack(canvases)).to(detector.device)
    outputs = detector.model(batch).float().cpu()
    results = []
    for image, letterbox, frame in zip(
        decode_detections(outputs, score_threshold, NMS_IOU), letterboxes, frames, strict=True
    ):
        boxes = letterbox.boxes_to_frame(image.boxes.numpy().astype(np.float64))
        h, w = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        results.append(
            FrameDetections(boxes=boxes[keep], scores=image.scores.numpy().astype(np.float64)[keep])
        )
    return results


def detect_clip(detector: LoadedDetector, clip: Clip, batch_size: int = 8) -> list[FrameDetections]:
    """Detections for every frame of the clip's window, in frame order."""
    if clip.frame_dir is None:
        raise ValueError(f"clip {clip.name} has no frames")
    frame_dir = clip.frame_dir

    def frames() -> Iterator[NDArray[np.uint8]]:
        for i in range(clip.frame_start, clip.frame_stop):
            gray = cv2.imread(str(frame_path(frame_dir, i)), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"cannot decode frame {i} of {clip.name}")
            yield np.asarray(gray, dtype=np.uint8)

    results: list[FrameDetections] = []
    batch: list[NDArray[np.uint8]] = []
    for image in encode_frames(detector.preprocessing, frames):
        batch.append(image)
        if len(batch) == batch_size:
            results.extend(detect_frames(detector, batch))
            batch = []
    if batch:
        results.extend(detect_frames(detector, batch))
    return results


def save_detections(path: Path, frame_start: int, detections: list[FrameDetections]) -> None:
    counts = np.array([len(d.scores) for d in detections], dtype=np.int64)
    boxes = np.concatenate([d.boxes for d in detections]) if counts.sum() else np.empty((0, 4))
    scores = np.concatenate([d.scores for d in detections]) if counts.sum() else np.empty(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, frame_start=frame_start, counts=counts, boxes=boxes, scores=scores)
    tmp.replace(path)


def load_detections(path: Path) -> tuple[int, list[FrameDetections]]:
    with np.load(path) as data:
        counts, boxes, scores = data["counts"], data["boxes"], data["scores"]
        frame_start = int(data["frame_start"])
    edges = np.concatenate([[0], np.cumsum(counts)])
    return frame_start, [
        FrameDetections(boxes=boxes[a:b], scores=scores[a:b])
        for a, b in itertools.pairwise(edges.tolist())
    ]


def above(detections: list[FrameDetections], threshold: float) -> list[FrameDetections]:
    return [
        FrameDetections(d.boxes[d.scores >= threshold], d.scores[d.scores >= threshold])
        for d in detections
    ]


@dataclass(frozen=True)
class ClipEvaluation:
    error: ClipCountError
    detection: DetectionMatch
    tracks: int


def track_clip(
    clip: Clip, detections: list[FrameDetections], tracker: TrackerConfig | ByteTrackConfig
) -> list[Trajectory]:
    """Track ``detections`` (one per window frame). The Kalman tracker associates by distance
    in meters (from the clip's meter extents); ByteTrack by box IoU."""
    if isinstance(tracker, ByteTrackConfig):
        return ByteTracker(tracker).run(detections, frame_offset=clip.frame_start)
    meta = clip.metadata
    scale = Scale(
        sx=1.0,
        sy=1.0,
        meters_per_px_x=abs(meta.x_meter_stop - meta.x_meter_start) / meta.width,
        meters_per_px_y=abs(meta.y_meter_stop - meta.y_meter_start) / meta.height,
    )
    centers = [boxes_in_meters(d.boxes, scale) for d in detections]
    return KalmanTracker(tracker).run(detections, centers, frame_offset=clip.frame_start)


def track_and_evaluate(
    clip: Clip,
    detections: list[FrameDetections],
    tracker: TrackerConfig | ByteTrackConfig,
    policy: CountingPolicy = CFC_COMPATIBLE_V1,
) -> ClipEvaluation:
    """Track ``detections`` (one per window frame) and compare counts with the reference.

    Any detector's output can be evaluated this way, with either tracker and the same
    counting.
    """
    meta = clip.metadata
    trajectories = track_clip(clip, detections, tracker)
    tracks = trajectories_to_annotations(trajectories)
    error = ClipCountError(
        clip.name,
        count_clip(clip.annotations, meta, policy),
        count_clip(tracks, meta, policy),
        clip.num_window_frames / meta.framerate,
    )
    return ClipEvaluation(
        error=error,
        detection=match_detections(clip.annotations, detections, clip.frame_start),
        tracks=len(trajectories),
    )
