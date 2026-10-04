"""Evaluating a declared release on the official test locations (once) or on the holdout.

Nothing is selected here. The release is fixed by its committed manifest, and the bundle
evaluated must match that manifest exactly. The test locations' archives interleave frames
from all clips, so a location's frames are streamed in **chunks of whole recording days**
that fit a disk budget (:func:`plan_day_chunks`), evaluated, and deleted. Clips whose frames
are incomplete are quarantined with a recorded reason, never silently skipped.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from passagewatch.ingestion.cfc import FRAME_SUFFIX
from passagewatch.ingestion.manifest import read_manifest
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.splits import HOLDOUT, Split, parse_clip_name

RELEASE_PARTITIONS = (Split.TEST.value, HOLDOUT)


def plan_day_chunks(
    clips: Mapping[str, ClipMetadata], bytes_per_frame: float, budget_bytes: float
) -> list[tuple[dt.date, ...]]:
    """Recording days in date order, grouped so each group's frames fit ``budget_bytes``
    (a single day larger than the budget gets a group of its own)."""
    frames_per_day: dict[dt.date, int] = {}
    for name, meta in clips.items():
        day = parse_clip_name(name).recording_date
        frames_per_day[day] = frames_per_day.get(day, 0) + meta.num_frames
    chunks: list[tuple[dt.date, ...]] = []
    current: list[dt.date] = []
    size = 0.0
    for day in sorted(frames_per_day):
        day_bytes = frames_per_day[day] * bytes_per_frame
        if current and size + day_bytes > budget_bytes:
            chunks.append(tuple(current))
            current, size = [], 0.0
        current.append(day)
        size += day_bytes
    if current:
        chunks.append(tuple(current))
    return chunks


def release_rows(manifest_path: Path, partition: str, location: str) -> list[dict[str, Any]]:
    """Usable manifest rows of ``location`` in ``partition`` (test or holdout only).

    This deliberately bypasses the development guard of ``select_rows``: a declared release
    is evaluated on the test partition exactly once, with nothing chosen on it.
    """
    if partition not in RELEASE_PARTITIONS:
        raise ValueError(f"releases are evaluated on {RELEASE_PARTITIONS}, not {partition!r}")
    return [
        r
        for r in read_manifest(manifest_path)
        if r["partition"] == partition and r["location"] == location and r["usable"]
    ]


def frames_problem(frame_dir: Path, num_frames: int) -> str | None:
    """Why a clip's frames cannot be evaluated, or ``None`` if ``0..num_frames-1`` exist."""
    if not frame_dir.is_dir():
        return "no_frames"
    names = {p.name for p in frame_dir.iterdir()}
    missing = [i for i in range(num_frames) if f"{i}{FRAME_SUFFIX}" not in names]
    if missing:
        return f"missing_frames:{len(missing)}"
    return None
