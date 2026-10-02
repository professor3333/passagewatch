"""Human review: append-only correction events, each producing a new result revision.

Revision 0 is the automatic result and never changes. A review event names the revision it
was based on; if the job has moved on since (another reviewer saved first), the event is
refused with :class:`StaleRevisionError` (HTTP 409). Otherwise the event is stored in
``review_events`` and a new ``result_revisions`` row records the decisions so far and the
reviewed counts, both in one transaction.

Per-track review states and how a track contributes to the reviewed counts:

| State | Contribution |
|---|---|
| automatic (not reviewed) | its automatic direction |
| ``accepted`` | its automatic direction (confirmed) |
| ``rejected`` | nothing (not a fish, or a duplicate fragment) |
| ``corrected`` | the reviewer's direction (``right``, ``left``, or none) |
| ``unresolved`` | nothing; listed separately as unresolved |

A reviewer can also add a passage the model missed entirely (``add_passage``), with the
frame where it is visible as evidence; added passages count in the reviewed counts and can be
rejected like tracks. Two fragments of one fish are fixed by rejecting one and correcting the
other.

Random audit windows (``audit.json``, when the release has a calibration version) are marked
as checked with ``mark_audited``; a passage found while watching one is added with
``add_passage`` and that window's ``audit_window`` index, so the export can say how many
passages audits found. Marking a window does not change any count.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from passagewatch.service.db import transaction
from passagewatch.service.jobs import canonical_json, iso

Action = Literal[
    "accept", "reject", "set_direction", "mark_unresolved", "add_passage", "mark_audited"
]
Direction = Literal["right", "left"]


class StaleRevisionError(RuntimeError):
    """The review was based on an older revision than the job's latest."""


class InvalidReviewError(ValueError):
    """The review refers to something that does not exist or is inconsistent."""


@dataclass(frozen=True)
class AutomaticTrack:
    track_id: int
    direction: str | None  # "right", "left" or None (no passage)


@dataclass(frozen=True)
class ReviewEvent:
    base_revision: int
    action: Action
    track_id: int | None = None
    passage_id: str | None = None
    direction: Direction | None = None
    frame_index: int | None = None
    audit_window: int | None = None
    reason: str = ""


def empty_decisions() -> dict[str, Any]:
    return {"tracks": {}, "passages": {}, "audits": {}}


def apply_event(
    decisions: dict[str, Any],
    event: ReviewEvent,
    automatic: dict[int, AutomaticTrack],
    num_frames: int,
    event_id: int,
    audit_windows: int = 0,
) -> dict[str, Any]:
    """Return the decisions after ``event``; raises :class:`InvalidReviewError`.
    ``audit_windows`` is the number of the job's audit windows."""
    new: dict[str, Any] = json.loads(json.dumps(decisions or empty_decisions()))
    new.setdefault("tracks", {})
    new.setdefault("passages", {})
    new.setdefault("audits", {})
    if event.audit_window is not None:
        if event.action not in ("add_passage", "mark_audited"):
            raise InvalidReviewError("audit_window goes with add_passage or mark_audited")
        if not 0 <= event.audit_window < audit_windows:
            raise InvalidReviewError(f"audit_window must be within [0, {audit_windows})")
    if event.action == "mark_audited":
        if event.audit_window is None:
            raise InvalidReviewError("mark_audited needs an audit_window")
        new["audits"][str(event.audit_window)] = {"state": "checked", "event_id": event_id}
        return new
    if event.action == "add_passage":
        if event.direction is None or event.frame_index is None:
            raise InvalidReviewError("add_passage needs a direction and a frame_index")
        if not 0 <= event.frame_index < num_frames:
            raise InvalidReviewError(f"frame_index must be within [0, {num_frames})")
        passage_id = f"p{event_id}"
        new["passages"][passage_id] = {
            "state": "added",
            "direction": event.direction,
            "frame_index": event.frame_index,
            "event_id": event_id,
        } | ({} if event.audit_window is None else {"audit_window": event.audit_window})
        return new

    if event.passage_id is not None:
        passage = new["passages"].get(event.passage_id)
        if passage is None:
            raise InvalidReviewError(f"unknown added passage {event.passage_id}")
        if event.action != "reject":
            raise InvalidReviewError("added passages can only be rejected")
        passage.update(state="rejected", event_id=event_id)
        return new

    if event.track_id is None or event.track_id not in automatic:
        raise InvalidReviewError(f"unknown track {event.track_id}")
    track = automatic[event.track_id]
    if event.action == "accept":
        decision = {"state": "accepted", "direction": track.direction}
    elif event.action == "reject":
        decision = {"state": "rejected", "direction": None}
    elif event.action == "set_direction":
        decision = {"state": "corrected", "direction": event.direction}
    elif event.action == "mark_unresolved":
        decision = {"state": "unresolved", "direction": None}
    else:  # pragma: no cover - guarded by the Literal type and the API schema
        raise InvalidReviewError(f"unknown action {event.action}")
    new["tracks"][str(event.track_id)] = decision | {"event_id": event_id}
    return new


def track_outcome(track: AutomaticTrack, decisions: dict[str, Any]) -> tuple[str, str | None]:
    """(review state, final direction) of one track under ``decisions``."""
    decision = decisions.get("tracks", {}).get(str(track.track_id))
    if decision is None:
        return "automatic", track.direction
    return decision["state"], decision["direction"]


def reviewed_counts(
    automatic: dict[int, AutomaticTrack], decisions: dict[str, Any]
) -> dict[str, int]:
    right = left = unresolved = reviewed = 0
    for track in automatic.values():
        state, direction = track_outcome(track, decisions)
        reviewed += state != "automatic"
        if state == "unresolved":
            unresolved += 1
        elif direction == "right":
            right += 1
        elif direction == "left":
            left += 1
    added = 0
    for passage in decisions.get("passages", {}).values():
        if passage["state"] == "added":
            added += 1
            right += passage["direction"] == "right"
            left += passage["direction"] == "left"
    return {
        "right": right,
        "left": left,
        "tracks": len(automatic),
        "reviewed_tracks": reviewed,
        "unresolved": unresolved,
        "added_passages": added,
    }


def submit_review(
    conn: sqlite3.Connection,
    job_id: str,
    event: ReviewEvent,
    *,
    automatic: dict[int, AutomaticTrack],
    num_frames: int,
    now: datetime,
    audit_windows: int = 0,
) -> int:
    """Store ``event`` and the revision it produces; returns the new revision number."""
    with transaction(conn):
        latest = conn.execute(
            "SELECT revision, tracks_artifact, decisions_json FROM result_revisions"
            " WHERE job_id = ? ORDER BY revision DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        if latest is None:
            raise InvalidReviewError(f"job {job_id} has no result to review")
        if event.base_revision != latest["revision"]:
            raise StaleRevisionError(
                f"review is based on revision {event.base_revision}, "
                f"but the latest is {latest['revision']}"
            )
        revision = int(latest["revision"]) + 1
        payload = {k: v for k, v in vars(event).items() if v is not None and k != "base_revision"}
        cursor = conn.execute(
            "INSERT INTO review_events (job_id, base_revision, resulting_revision, created_at,"
            " payload_json) VALUES (?, ?, ?, ?, ?)",
            (job_id, event.base_revision, revision, iso(now), canonical_json(payload)),
        )
        event_id = int(cursor.lastrowid or 0)
        decisions = apply_event(
            json.loads(latest["decisions_json"]),
            event,
            automatic,
            num_frames,
            event_id,
            audit_windows,
        )
        conn.execute(
            "INSERT INTO result_revisions (job_id, revision, kind, created_at, counts_json,"
            " tracks_artifact, review_event_id, decisions_json)"
            " VALUES (?, ?, 'reviewed', ?, ?, ?, ?, ?)",
            (
                job_id,
                revision,
                iso(now),
                canonical_json(reviewed_counts(automatic, decisions)),
                latest["tracks_artifact"],
                event_id,
                canonical_json(decisions),
            ),
        )
    return revision


def review_history(conn: sqlite3.Connection, job_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM review_events WHERE job_id = ? ORDER BY event_id", (job_id,)
    ).fetchall()
    return [
        {
            "event_id": r["event_id"],
            "created_at": r["created_at"],
            "base_revision": r["base_revision"],
            "resulting_revision": r["resulting_revision"],
            **json.loads(r["payload_json"]),
        }
        for r in rows
    ]


TRIAGE_RANK = {"unresolved": 0, "needs_review": 1, "suggested": 2}


def queue_key(row: dict[str, Any]) -> tuple[int, int, float, int]:
    """Review-queue order of a track row: unresolved, needs review, suggested (tracks
    without triage last); passages before other tracks; lowest review score first."""
    score = row.get("review_score")
    return (
        TRIAGE_RANK.get(row.get("triage") or "", len(TRIAGE_RANK)),
        0 if row["direction"] is not None else 1,
        float("inf") if score is None else float(score),
        int(row["track_id"]),
    )


def audit_view(audit: dict[str, Any] | None, decisions: dict[str, Any]) -> dict[str, Any] | None:
    """A job's audit windows with their review state under ``decisions``, and totals."""
    if audit is None:
        return None
    checked = decisions.get("audits", {})
    found: dict[int, int] = {}
    for p in decisions.get("passages", {}).values():
        if p["state"] == "added" and "audit_window" in p:
            found[p["audit_window"]] = found.get(p["audit_window"], 0) + 1
    windows = [
        w
        | {
            "state": "checked" if str(w["index"]) in checked else "pending",
            "passages_added": found.get(w["index"], 0),
        }
        for w in audit["windows"]
    ]
    done = [w for w in windows if w["state"] == "checked"]
    checked_frames = sum(w["stop_frame"] - w["start_frame"] for w in done)
    return {
        "calibration_version": audit["calibration_version"],
        "unflagged_frames": audit["unflagged_frames"],
        "windows": windows,
        "windows_checked": len(done),
        "checked_frames": checked_frames,
        "checked_fraction_of_unflagged": (
            checked_frames / audit["unflagged_frames"] if audit["unflagged_frames"] else None
        ),
        "passages_added_in_audits": sum(found.values()),
    }
