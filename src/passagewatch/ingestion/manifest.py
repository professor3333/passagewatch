"""Versioned clip manifests: one Parquet row per clip, recording its partition and status.

A manifest version (for example ``tiny-v1``) is immutable. Writing a version that already
exists succeeds only if the content is identical; any change needs a new version name. Each
manifest has a JSON sidecar with its content hash, the hashes of its inputs, and per-partition
totals. The content hash covers the rows, not the Parquet bytes, so it does not depend on
the pyarrow version.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from passagewatch.ingestion.cfc import LOCATIONS, CfcLayout
from passagewatch.ingestion.inventory import sha256_of
from passagewatch.ingestion.splits import official_split, parse_clip_name, tuning_allowed
from passagewatch.validation.cfc import Severity, ValidationReport

SCHEMA_VERSION = 1
VERSION_PATTERN = re.compile(r"^(tiny|full)-v[1-9][0-9]*$")
SPLIT_RULE = (
    "partition = publisher split of the clip's location (kenai-train -> train, "
    "kenai-val -> val, all other locations -> test); whole clips only; "
    "tuning_allowed = partition != test; quarantined clips are listed but not usable"
)

_FIELDS: list[pa.Field[Any]] = [
    pa.field("location", pa.string(), nullable=False),
    pa.field("clip_name", pa.string(), nullable=False),
    pa.field("recording_id", pa.string()),
    pa.field("recording_date", pa.date32()),
    pa.field("recording_frame_start", pa.int64()),
    pa.field("recording_frame_stop", pa.int64()),
    pa.field("official_split", pa.string(), nullable=False),
    pa.field("partition", pa.string(), nullable=False),
    pa.field("tuning_allowed", pa.bool_(), nullable=False),
    pa.field("status", pa.string(), nullable=False),
    pa.field("usable", pa.bool_(), nullable=False),
    pa.field("quarantine_reasons", pa.list_(pa.string()), nullable=False),
    pa.field("warning_codes", pa.list_(pa.string()), nullable=False),
    pa.field("frames_validated", pa.bool_(), nullable=False),
    pa.field("num_frames", pa.int64()),
    pa.field("frame_start", pa.int64()),
    pa.field("frame_stop", pa.int64()),
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    pa.field("framerate", pa.float64()),
    pa.field("boxes", pa.int64()),
    pa.field("tracks", pa.int64()),
    pa.field("gt_sha256", pa.string()),
]
MANIFEST_SCHEMA = pa.schema(_FIELDS)


class ManifestVersionError(ValueError):
    """A manifest version already exists with different content."""


def build_rows(layout: CfcLayout, report: ValidationReport) -> list[dict[str, Any]]:
    """One row per clip in the validation report, ordered by location then clip name."""
    order = {loc: i for i, loc in enumerate(LOCATIONS)}
    metadata = {loc: layout.metadata(loc).clips for loc in {c.location for c in report.clips}}
    rows: list[dict[str, Any]] = []
    for clip in sorted(report.clips, key=lambda c: (order[c.location], c.clip_name)):
        split = official_split(clip.location)
        try:
            name = parse_clip_name(clip.clip_name)
        except ValueError:
            name = None
        meta = metadata[clip.location].get(clip.clip_name)
        stats = clip.stats
        gt = layout.gt_path(clip.location, clip.clip_name)
        rows.append(
            {
                "location": clip.location,
                "clip_name": clip.clip_name,
                "recording_id": name.recording_id if name else None,
                "recording_date": name.recording_date if name else None,
                "recording_frame_start": name.start if name else None,
                "recording_frame_stop": name.stop if name else None,
                "official_split": split.value,
                "partition": split.value,
                "tuning_allowed": tuning_allowed(split),
                "status": clip.status,
                "usable": clip.status != "quarantined",
                "quarantine_reasons": clip.quarantine_reasons,
                "warning_codes": sorted(
                    {i.code.value for i in clip.issues if i.severity is Severity.WARNING}
                ),
                "frames_validated": bool(stats and stats.frames_checked),
                "num_frames": meta.num_frames if meta else None,
                "frame_start": stats.frame_start if stats else None,
                "frame_stop": stats.frame_stop if stats else None,
                "width": meta.width if meta else None,
                "height": meta.height if meta else None,
                "framerate": meta.framerate if meta else None,
                "boxes": stats.boxes if stats else None,
                "tracks": stats.tracks if stats else None,
                "gt_sha256": sha256_of(gt) if gt.is_file() else None,
            }
        )
    return rows


def content_sha256(rows: list[dict[str, Any]]) -> str:
    def default(value: object) -> str:
        if isinstance(value, dt.date):
            return value.isoformat()
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, default=default).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        t = totals.setdefault(
            row["partition"], {"clips": 0, "usable_clips": 0, "window_frames": 0, "boxes": 0}
        )
        t["clips"] += 1
        if row["usable"]:
            t["usable_clips"] += 1
            t["window_frames"] += row["frame_stop"] - row["frame_start"]
            t["boxes"] += row["boxes"]
    return dict(sorted(totals.items()))


@dataclass(frozen=True)
class ManifestPaths:
    parquet: Path
    sidecar: Path


def manifest_paths(out_dir: Path, version: str) -> ManifestPaths:
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"manifest version must look like 'tiny-v1' or 'full-v2': {version!r}")
    return ManifestPaths(out_dir / f"{version}.parquet", out_dir / f"{version}.json")


def write_manifest(
    rows: list[dict[str, Any]],
    out_dir: Path,
    version: str,
    *,
    inputs: dict[str, str],
) -> ManifestPaths:
    """Write ``rows`` as manifest ``version``; refuse to change an existing version.

    ``inputs`` maps input names (validation report, inventories) to their SHA-256.
    """
    paths = manifest_paths(out_dir, version)
    digest = content_sha256(rows)
    if paths.sidecar.exists():
        existing = json.loads(paths.sidecar.read_text(encoding="utf-8"))
        if existing["content_sha256"] != digest or existing["inputs"] != inputs:
            raise ManifestVersionError(
                f"manifest {version} already exists with different content or inputs; "
                "write a new version instead"
            )
        if paths.parquet.exists():
            return paths

    sidecar = {
        "name": f"cfc-{version}",
        "schema_version": SCHEMA_VERSION,
        "dataset": "cfc",
        "subset": version.split("-")[0],
        "rows": len(rows),
        "content_sha256": digest,
        "split_rule": SPLIT_RULE,
        "inputs": inputs,
        "partitions": _totals(rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
    tmp = paths.parquet.with_name(paths.parquet.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(paths.parquet)
    tmp = paths.sidecar.with_name(paths.sidecar.name + ".tmp")
    tmp.write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")
    tmp.replace(paths.sidecar)
    return paths


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read a manifest and check it against its sidecar's content hash."""
    rows = pq.read_table(path, schema=MANIFEST_SCHEMA).to_pylist()
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if content_sha256(rows) != sidecar["content_sha256"]:
        raise ManifestVersionError(f"{path} does not match its sidecar content hash")
    return rows
