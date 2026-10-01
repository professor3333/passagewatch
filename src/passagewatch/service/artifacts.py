"""Per-job track artifacts: compressed Parquet files, not database rows.

``<artifacts>/<job_id>/tracks.parquet`` has one row per trajectory (the counting outcome and
the evidence a reviewer needs); ``observations.parquet`` has every box of every trajectory,
for overlays. Files are written to a temporary name and renamed, so a reader never sees a
partial artifact, and a retried job rewrites them identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from passagewatch.counting.policy import TrajectoryCount
from passagewatch.tracking.kalman import Trajectory

TRACKS_FILE = "tracks.parquet"
OBSERVATIONS_FILE = "observations.parquet"

_TRACK_FIELDS: list[pa.Field[Any]] = [
    pa.field("track_id", pa.int64(), nullable=False),
    pa.field("start_frame", pa.int64(), nullable=False),
    pa.field("end_frame", pa.int64(), nullable=False),
    pa.field("start_time_s", pa.float64(), nullable=False),
    pa.field("end_time_s", pa.float64(), nullable=False),
    pa.field("observations", pa.int64(), nullable=False),
    pa.field("start_u", pa.float64(), nullable=False),
    pa.field("end_u", pa.float64(), nullable=False),
    pa.field("displacement", pa.float64(), nullable=False),
    pa.field("outcome", pa.string(), nullable=False),
    pa.field("direction", pa.string()),
    pa.field("mean_score", pa.float64(), nullable=False),
    pa.field("min_score", pa.float64(), nullable=False),
]
TRACKS_SCHEMA = pa.schema(_TRACK_FIELDS)

_OBSERVATION_FIELDS: list[pa.Field[Any]] = [
    pa.field("track_id", pa.int64(), nullable=False),
    pa.field("frame_index", pa.int64(), nullable=False),
    pa.field("x_min", pa.float64(), nullable=False),
    pa.field("y_min", pa.float64(), nullable=False),
    pa.field("x_max", pa.float64(), nullable=False),
    pa.field("y_max", pa.float64(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
]
OBSERVATIONS_SCHEMA = pa.schema(_OBSERVATION_FIELDS)


def _write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def write_track_artifacts(
    job_dir: Path,
    trajectories: list[Trajectory],
    counts: list[TrajectoryCount],
    framerate: float,
) -> None:
    """``counts[i]`` must describe ``trajectories[i]`` (same track IDs, same order)."""
    if [t.track_id for t in trajectories] != [c.track_id for c in counts]:
        raise ValueError("trajectories and counts do not describe the same tracks")
    rows = [
        {
            "track_id": c.track_id,
            "start_frame": c.start_frame,
            "end_frame": c.end_frame,
            "start_time_s": c.start_frame / framerate,
            "end_time_s": c.end_frame / framerate,
            "observations": c.observations,
            "start_u": c.start_u,
            "end_u": c.end_u,
            "displacement": c.displacement,
            "outcome": c.outcome.value,
            "direction": None if c.direction is None else c.direction.value,
            "mean_score": float(np.mean(t.scores)),
            "min_score": float(np.min(t.scores)),
        }
        for t, c in zip(trajectories, counts, strict=True)
    ]
    _write(pa.Table.from_pylist(rows, schema=TRACKS_SCHEMA), job_dir / TRACKS_FILE)
    observations = {name: [] for name in OBSERVATIONS_SCHEMA.names}  # type: ignore[var-annotated]
    for t in trajectories:
        observations["track_id"].extend([t.track_id] * len(t.frames))
        observations["frame_index"].extend(t.frames.tolist())
        for i, name in enumerate(("x_min", "y_min", "x_max", "y_max")):
            observations[name].extend(t.boxes[:, i].tolist())
        observations["score"].extend(t.scores.tolist())
    _write(pa.table(observations, schema=OBSERVATIONS_SCHEMA), job_dir / OBSERVATIONS_FILE)


@dataclass(frozen=True)
class TrackPage:
    total: int
    tracks: list[dict[str, Any]]


def read_tracks(job_dir: Path, *, offset: int, limit: int) -> TrackPage:
    table = pq.read_table(job_dir / TRACKS_FILE, schema=TRACKS_SCHEMA)
    rows = table.slice(offset, limit).to_pylist()
    return TrackPage(total=table.num_rows, tracks=rows)
