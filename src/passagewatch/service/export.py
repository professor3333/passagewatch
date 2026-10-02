"""Reports: everything needed to trace a count back to its recording, model, and review.

A report is built for one result revision (the latest by default) and always contains the
recording identity, the counting configuration (line and orientation), the automatic counts
and, after review, the reviewed counts, the unresolved cases, the model and pipeline
versions, and evidence frames and times for every track and added passage. JSON keeps the
full structure; CSV puts the same metadata and counts in ``#`` header lines above one row
per track and added passage. With a calibration version, every track also carries its
heuristic review score and triage state, and the report lists the random audit windows with
which ones a reviewer checked and the passages found in them.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from pathlib import Path
from typing import Any

from passagewatch.counting.policy import Direction, DirectionalCounts, to_river_directions
from passagewatch.service.artifacts import read_all_tracks, read_audit
from passagewatch.service.catalog import get_clip, get_pipeline_version
from passagewatch.service.jobs import JobStore, iso, utc_now
from passagewatch.service.reviews import (
    AutomaticTrack,
    audit_view,
    review_history,
    track_outcome,
)

REPORT_VERSION = 2


def automatic_tracks(rows: list[dict[str, Any]]) -> dict[int, AutomaticTrack]:
    return {int(r["track_id"]): AutomaticTrack(int(r["track_id"]), r["direction"]) for r in rows}


def river_direction(direction: str | None, upstream: str | None) -> str | None:
    if direction is None or upstream is None:
        return None
    return "upstream" if direction == upstream else "downstream"


def counts_view(counts: dict[str, Any], upstream: str | None) -> dict[str, Any]:
    orientation = None if upstream is None else Direction(upstream)
    river = to_river_directions(DirectionalCounts(counts["right"], counts["left"]), orientation)
    return {
        "right": counts["right"],
        "left": counts["left"],
        "upstream": None if river is None else river.upstream,
        "downstream": None if river is None else river.downstream,
        "net_upstream": None if river is None else river.net,
        **{k: v for k, v in counts.items() if k not in ("right", "left")},
    }


def build_report(
    conn: sqlite3.Connection, artifacts_dir: Path, job_id: str, revision: int | None = None
) -> dict[str, Any]:
    store = JobStore(conn)
    job = store.get(job_id)
    if job is None:
        raise KeyError(job_id)
    chosen = store.result(job_id, revision)
    automatic = store.result(job_id, 0)
    if chosen is None or automatic is None:
        raise KeyError(f"{job_id} revision {revision}")
    clip = get_clip(conn, job.clip_id)
    pipeline = get_pipeline_version(conn, job.pipeline_version)
    assert clip is not None and pipeline is not None
    upstream = job.counting.get("upstream_direction")
    rows = read_all_tracks(artifacts_dir / chosen.tracks_artifact)
    tracks = automatic_tracks(rows)

    track_rows = []
    unresolved = []
    for row in rows:
        state, final = track_outcome(tracks[int(row["track_id"])], chosen.decisions)
        entry = {
            "track_id": row["track_id"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "start_time_s": row["start_time_s"],
            "end_time_s": row["end_time_s"],
            "automatic_direction": row["direction"],
            "review_state": state,
            "final_direction": final,
            "final_river_direction": river_direction(final, upstream),
            "mean_score": row["mean_score"],
            "triage": row["triage"],
            "review_score": row["review_score"],
            "review_reasons": row["review_reasons"],
        }
        track_rows.append(entry)
        if state == "unresolved":
            unresolved.append(entry)
    passages = [
        {
            "passage_id": passage_id,
            "state": p["state"],
            "direction": p["direction"],
            "river_direction": river_direction(p["direction"], upstream),
            "frame_index": p["frame_index"],
            "time_s": p["frame_index"] / clip.framerate,
        }
        for passage_id, p in sorted(chosen.decisions.get("passages", {}).items())
    ]
    history = [
        e for e in review_history(conn, job_id) if e["resulting_revision"] <= chosen.revision
    ]
    return {
        "report_version": REPORT_VERSION,
        "generated_at": iso(utc_now()),
        "recording": {
            "clip_id": clip.clip_id,
            "sha256": clip.sha256,
            "media_kind": clip.media_kind,
            "num_frames": clip.num_frames,
            "framerate": clip.framerate,
            "duration_seconds": clip.duration_seconds,
            "meters": {
                "x_start": clip.x_meter_start,
                "x_stop": clip.x_meter_stop,
                "y_start": clip.y_meter_start,
                "y_stop": clip.y_meter_stop,
            },
        },
        "counting": job.counting,
        "pipeline": {
            "pipeline_version": job.pipeline_version,
            "config_sha256": pipeline.config_sha256,
            **pipeline.config,
        },
        "job": {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "cached_from": job.cached_from,
        },
        "revision": chosen.revision,
        "revision_kind": chosen.kind,
        "counts": {
            "automatic": counts_view(automatic.counts, upstream),
            "reviewed": None if chosen.revision == 0 else counts_view(chosen.counts, upstream),
        },
        "unresolved": unresolved,
        "tracks": track_rows,
        "added_passages": passages,
        "audit": audit_view(read_audit(artifacts_dir / chosen.tracks_artifact), chosen.decisions),
        "review_events": history,
    }


CSV_COLUMNS = [
    "kind",
    "id",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "automatic_direction",
    "review_state",
    "final_direction",
    "final_river_direction",
    "triage",
    "review_score",
]


def to_csv(report: dict[str, Any]) -> str:
    out = io.StringIO()
    rec, counting, pipeline = report["recording"], report["counting"], report["pipeline"]
    meta = [
        ("report_version", report["report_version"]),
        ("generated_at", report["generated_at"]),
        ("recording_id", rec["clip_id"]),
        ("recording_sha256", rec["sha256"]),
        ("counting_policy", counting["policy"]),
        ("line_x_normalized", counting["line_x_normalized"]),
        ("upstream_direction", counting["upstream_direction"] or "not set"),
        ("pipeline_version", pipeline["pipeline_version"]),
        ("pipeline_config_sha256", pipeline["config_sha256"]),
        ("detector_checkpoint_sha256", pipeline["detector"]["checkpoint_sha256"]),
        ("preprocessing_version", pipeline["preprocessing_version"]),
        ("calibration_version", pipeline.get("calibration_version") or "none"),
        ("revision", report["revision"]),
        ("revision_kind", report["revision_kind"]),
    ]
    audit = report["audit"]
    if audit is not None:
        meta += [
            ("audit_windows", len(audit["windows"])),
            ("audit_windows_checked", audit["windows_checked"]),
            ("audit_passages_added", audit["passages_added_in_audits"]),
        ]
    for label in ("automatic", "reviewed"):
        counts = report["counts"][label]
        for key in ("right", "left", "upstream", "downstream", "net_upstream", "unresolved"):
            value = None if counts is None else counts.get(key)
            meta.append((f"{label}_{key}", "" if value is None else value))
    for key, value in meta:
        out.write(f"# {key},{value}\n")
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for t in report["tracks"]:
        writer.writerow(
            {
                "kind": "track",
                "id": t["track_id"],
                **{k: t[k] for k in CSV_COLUMNS[2:] if k in t},
            }
        )
    for p in report["added_passages"]:
        writer.writerow(
            {
                "kind": "added_passage",
                "id": p["passage_id"],
                "start_frame": p["frame_index"],
                "end_frame": p["frame_index"],
                "start_time_s": p["time_s"],
                "end_time_s": p["time_s"],
                "automatic_direction": "",
                "review_state": p["state"],
                "final_direction": p["direction"] if p["state"] == "added" else "",
                "final_river_direction": (p["river_direction"] or "")
                if p["state"] == "added"
                else "",
                "triage": "",
                "review_score": "",
            }
        )
    return out.getvalue()
