"""Run the classical pipeline over many clips, in parallel, and evaluate every clip.

Each clip is independent (track IDs never cross recordings), so clips run in separate
processes. Results come back as plain data and are written by the caller.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from passagewatch.counting.policy import CFC_COMPATIBLE_V1, CountingPolicy
from passagewatch.evaluation.detection import DetectionMatch, match_detections
from passagewatch.evaluation.nmae import (
    ClipCountError,
    GroupResult,
    count_clip,
    macro_nmae,
    summarize,
)
from passagewatch.inference.classical import ClassicalConfig, run_clip
from passagewatch.ingestion.cfc import CfcLayout, load_clip
from passagewatch.ingestion.manifest import read_manifest
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.mot import write_mot

TRACKER = "classical"


@dataclass(frozen=True)
class ClipTask:
    layout: CfcLayout
    location: str
    partition: str
    metadata: ClipMetadata
    config: ClassicalConfig
    policy: CountingPolicy
    tracks_dir: Path | None


@dataclass(frozen=True)
class ClipOutcome:
    location: str
    partition: str
    error: ClipCountError
    detection: DetectionMatch
    frames: int
    tracks: int
    seconds: dict[str, float]


def run_task(task: ClipTask) -> ClipOutcome:
    clip = load_clip(task.layout, task.location, task.metadata)
    result = run_clip(clip, task.config)
    if task.tracks_dir is not None:
        path = task.tracks_dir / task.location / TRACKER / "data" / f"{clip.name}.txt"
        write_mot(result.tracks, path)
    error = ClipCountError(
        clip.name,
        count_clip(clip.annotations, task.metadata, task.policy),
        count_clip(result.tracks, task.metadata, task.policy),
        clip.num_window_frames / task.metadata.framerate,
    )
    return ClipOutcome(
        location=task.location,
        partition=task.partition,
        error=error,
        detection=match_detections(clip.annotations, result.frame_detections, clip.frame_start),
        frames=clip.num_window_frames,
        tracks=len(result.trajectories),
        seconds=result.seconds,
    )


def run_tasks(
    tasks: Sequence[ClipTask],
    workers: int = 1,
    on_done: Callable[[int, ClipOutcome], None] | None = None,
) -> list[ClipOutcome]:
    """Run ``tasks``; results keep the order of ``tasks``."""
    if workers <= 1:
        outcomes = []
        for i, task in enumerate(tasks, 1):
            outcomes.append(run_task(task))
            if on_done:
                on_done(i, outcomes[-1])
        return outcomes
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = []
        for i, outcome in enumerate(pool.map(run_task, tasks, chunksize=1), 1):
            outcomes.append(outcome)
            if on_done:
                on_done(i, outcome)
        return outcomes


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def make_tasks(
    rows: Sequence[dict[str, Any]],
    layout: CfcLayout,
    config: ClassicalConfig,
    tracks_dir: Path | None,
    policy: CountingPolicy = CFC_COMPATIBLE_V1,
) -> list[ClipTask]:
    metadata = {loc: layout.metadata(loc).clips for loc in {r["location"] for r in rows}}
    return [
        ClipTask(
            layout=layout,
            location=r["location"],
            partition=r["partition"],
            metadata=metadata[r["location"]][r["clip_name"]],
            config=config,
            policy=policy,
            tracks_dir=tracks_dir,
        )
        for r in rows
    ]


def summarize_outcomes(outcomes: Sequence[ClipOutcome]) -> dict[str, Any]:
    """Per-location counting and detection results, plus totals and runtime."""
    by_location: dict[str, list[ClipOutcome]] = {}
    for outcome in outcomes:
        by_location.setdefault(outcome.location, []).append(outcome)
    groups: list[GroupResult] = []
    locations = []
    for location, items in by_location.items():
        group = summarize(location, [o.error for o in items])
        groups.append(group)
        detection = sum((o.detection for o in items), DetectionMatch(0, 0, 0))
        locations.append(
            {
                "location": location,
                "clips": group.clips,
                "absolute_error": group.absolute_error,
                "reference_passages": group.reference_passages,
                "predicted_passages": group.predicted_passages,
                "nmae": group.nmae,
                "detection_recall": detection.recall,
                "detection_precision": detection.precision,
            }
        )
    seconds: dict[str, float] = {}
    for outcome in outcomes:
        for key, value in outcome.seconds.items():
            seconds[key] = seconds.get(key, 0.0) + value
    frames = sum(o.frames for o in outcomes)
    return {
        "locations": locations,
        "macro_nmae": macro_nmae(groups),
        "runtime": {
            "frames": frames,
            "cpu_seconds": {k: round(v, 2) for k, v in seconds.items()},
            "ms_per_frame": round(1000 * sum(seconds.values()) / max(1, frames), 2),
        },
    }


def clip_records(outcomes: Sequence[ClipOutcome]) -> list[dict[str, Any]]:
    return [
        {
            "location": o.location,
            "partition": o.partition,
            "clip_name": o.error.clip_name,
            "frames": o.frames,
            "tracks": o.tracks,
            "reference": [o.error.reference.right, o.error.reference.left],
            "predicted": [o.error.predicted.right, o.error.predicted.left],
            "absolute_error": o.error.absolute_error,
            "detection": [o.detection.matched, o.detection.reference_boxes, o.detection.detections],
        }
        for o in outcomes
    ]


def select_rows(
    manifest_path: Path, partitions: Sequence[str], limit: int | None = None
) -> list[dict[str, Any]]:
    """Usable manifest rows with validated frames in ``partitions`` (never test)."""
    if "test" in partitions:
        raise ValueError("the test partition cannot be run for development or tuning")
    rows = [
        r
        for r in read_manifest(manifest_path)
        if r["partition"] in partitions and r["usable"] and r["frames_validated"]
    ]
    if not all(r["tuning_allowed"] for r in rows):
        raise RuntimeError("selected rows include clips that are not allowed for tuning")
    return rows[:limit] if limit else rows


def make_layout(
    subset: str, extract_dir: Path, frames_dir: Path | None, rows: Sequence[dict[str, Any]]
) -> CfcLayout:
    """The layout for a manifest's subset; full manifests need the frames directory."""
    if subset == "tiny":
        return CfcLayout.tiny(extract_dir)
    if frames_dir is None:
        raise ValueError("--frames-dir is required for full manifests")
    return CfcLayout.full(extract_dir, frames_dir, frozenset(r["clip_name"] for r in rows))
