"""Per-job track artifacts: compressed Parquet files, not database rows.

``<artifacts>/<job_id>/tracks.parquet`` has one row per trajectory (the counting outcome and
the evidence a reviewer needs); ``observations.parquet`` has every box of every trajectory,
for overlays. Files are written to a temporary name and renamed, so a reader never sees a
partial artifact, and a retried job rewrites them identically.

When the release declares a calibration version, each track also has its review score,
triage state and reasons, and ``audit.json`` lists the job's random audit windows. Artifacts
written without one (or before these columns existed) read back with those fields null.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from passagewatch.calibration.review_score import TrackReview
from passagewatch.calibration.versions import ClipReview
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
    pa.field("review_score", pa.float64()),
    pa.field("triage", pa.string()),
    pa.field("review_reasons", pa.list_(pa.string())),
]
REVIEW_COLUMNS = ("review_score", "triage", "review_reasons")
AUDIT_FILE = "audit.json"
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
    review: ClipReview | None = None,
    calibration_version: str | None = None,
) -> None:
    """``counts[i]`` must describe ``trajectories[i]`` (same track IDs, same order)."""
    if [t.track_id for t in trajectories] != [c.track_id for c in counts]:
        raise ValueError("trajectories and counts do not describe the same tracks")
    reviews = review.tracks if review is not None else {}
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
            **_review_columns(reviews.get(c.track_id)),
        }
        for t, c in zip(trajectories, counts, strict=True)
    ]
    _write(pa.Table.from_pylist(rows, schema=TRACKS_SCHEMA), job_dir / TRACKS_FILE)
    if review is not None:
        audit = {
            "calibration_version": calibration_version,
            "unflagged_frames": review.unflagged_frames,
            "windows": [
                {
                    "index": i,
                    "start_frame": w.start_frame,
                    "stop_frame": w.stop_frame,
                    "start_time_s": w.start_frame / framerate,
                    "stop_time_s": w.stop_frame / framerate,
                }
                for i, w in enumerate(review.audit)
            ],
        }
        path = job_dir / AUDIT_FILE
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)
    observations = {name: [] for name in OBSERVATIONS_SCHEMA.names}  # type: ignore[var-annotated]
    for t in trajectories:
        observations["track_id"].extend([t.track_id] * len(t.frames))
        observations["frame_index"].extend(t.frames.tolist())
        for i, name in enumerate(("x_min", "y_min", "x_max", "y_max")):
            observations[name].extend(t.boxes[:, i].tolist())
        observations["score"].extend(t.scores.tolist())
    _write(pa.table(observations, schema=OBSERVATIONS_SCHEMA), job_dir / OBSERVATIONS_FILE)


def _review_columns(r: TrackReview | None) -> dict[str, Any]:
    if r is None:
        return {"review_score": None, "triage": None, "review_reasons": None}
    return {
        "review_score": r.review_score,
        "triage": r.triage,
        "review_reasons": list(r.reasons),
    }


def _read_tracks_table(job_dir: Path) -> pa.Table:
    """The tracks table, with null review columns if the artifact predates them."""
    table = pq.read_table(job_dir / TRACKS_FILE)
    for name in REVIEW_COLUMNS:
        if name not in table.column_names:
            field = TRACKS_SCHEMA.field(name)
            table = table.append_column(field, pa.nulls(table.num_rows, type=field.type))
    return table.select(TRACKS_SCHEMA.names).cast(TRACKS_SCHEMA)


def read_audit(job_dir: Path) -> dict[str, Any] | None:
    path = job_dir / AUDIT_FILE
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@dataclass(frozen=True)
class TrackPage:
    total: int
    tracks: list[dict[str, Any]]


def read_tracks(job_dir: Path, *, offset: int, limit: int) -> TrackPage:
    table = _read_tracks_table(job_dir)
    rows = table.slice(offset, limit).to_pylist()
    return TrackPage(total=table.num_rows, tracks=rows)


def read_all_tracks(job_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = _read_tracks_table(job_dir).to_pylist()
    return rows


def read_observations(job_dir: Path, *, start: int, stop: int) -> list[dict[str, Any]]:
    """Every box with ``start <= frame_index < stop``, ordered by frame then track."""
    table = pq.read_table(
        job_dir / OBSERVATIONS_FILE,
        schema=OBSERVATIONS_SCHEMA,
        filters=[("frame_index", ">=", start), ("frame_index", "<", stop)],
    )
    rows: list[dict[str, Any]] = table.sort_by(
        [("frame_index", "ascending"), ("track_id", "ascending")]
    ).to_pylist()
    return rows
