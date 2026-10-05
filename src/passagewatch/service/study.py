"""The usability study's plan and records (``docs/usability_study.md``).

Active only when the service runs with ``PASSAGEWATCH_STUDY_MODE`` and a plan
(``PASSAGEWATCH_STUDY_PLAN``, written by ``scripts/prepare_usability_study.py``). Records go
to their own database, ``<data_dir>/study.db``, never to the product's tables:

- ``trials``: one row per clip a participant finished (participant code, block, condition,
  clip, active milliseconds, pauses, final counts, and for assisted trials the job's review
  revision the counts came from);
- ``questionnaires``: SUS (10 items, 1-5) and NASA-TLX raw (6 scales, 0-100) per
  participant and condition.

The plan given to the browser leaves out the reference counts and the release's automatic
counts, so nothing on screen reveals the answers.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from passagewatch.service.db import transaction

Condition = Literal["manual", "assisted"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    participant   TEXT NOT NULL,
    block         INTEGER NOT NULL,
    condition     TEXT NOT NULL,
    clip          TEXT NOT NULL,
    practice      INTEGER NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT NOT NULL,
    active_ms     INTEGER NOT NULL,
    pauses        INTEGER NOT NULL,
    final_right   INTEGER NOT NULL,
    final_left    INTEGER NOT NULL,
    job_id        TEXT,
    revision      INTEGER,
    note          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (participant, clip)
);
CREATE TABLE IF NOT EXISTS questionnaires (
    participant   TEXT NOT NULL,
    condition     TEXT NOT NULL,
    sus           TEXT NOT NULL,
    tlx           TEXT NOT NULL,
    submitted_at  TEXT NOT NULL,
    PRIMARY KEY (participant, condition)
);
"""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StudyClip(_Frozen):
    clip_name: str
    clip_id: str
    role: Literal["practice", "S1", "S2"]
    camera: str
    num_frames: int
    framerate: float
    reference: tuple[int, int]
    automatic: tuple[int, int]


class StudyPlan(_Frozen):
    design: str
    release: str
    seed: int
    clips: dict[str, StudyClip]  # key: a short code (c01, c02, ...) shown to participants
    practice: list[str]  # the practice clip of block 1 and of block 2
    sets: dict[str, list[str]]  # S1/S2 -> clip keys, in presentation order
    participants: dict[str, list[dict[str, str]]]  # code -> blocks {condition, set}
    # participant -> clip key -> job: every assisted trial reviews its own copy of the
    # release's result, so nobody sees another participant's corrections
    jobs: dict[str, dict[str, str]]

    def blocks(self, participant: str) -> list[tuple[str, list[str]]]:
        """Each block's condition and clips, practice clip first."""
        return [
            (b["condition"], [self.practice[i], *self.sets[b["set"]]])
            for i, b in enumerate(self.participants[participant])
        ]


def load_plan(path: Path) -> StudyPlan:
    return StudyPlan.model_validate(json.loads(path.read_text(encoding="utf-8")))


def public_plan(plan: StudyPlan) -> dict[str, Any]:
    """The plan without reference or automatic counts, for the study pages."""
    return {
        "release": plan.release,
        "clips": {
            key: {
                "clip_id": c.clip_id,
                "role": c.role,
                "num_frames": c.num_frames,
                "framerate": c.framerate,
            }
            for key, c in plan.clips.items()
        },
        "practice": plan.practice,
        "sets": plan.sets,
        "participants": plan.participants,
        "jobs": plan.jobs,
    }


class TrialIn(_Frozen):
    participant: str = Field(pattern=r"^[A-Z][0-9]{1,3}$")
    block: int = Field(ge=1, le=2)
    condition: Condition
    clip: str
    started_at: str
    finished_at: str
    active_ms: int = Field(ge=0)
    pauses: int = Field(ge=0)
    right: int | None = Field(default=None, ge=0)  # manual trials only
    left: int | None = Field(default=None, ge=0)
    note: str = Field(default="", max_length=500)


class QuestionnaireIn(_Frozen):
    participant: str = Field(pattern=r"^[A-Z][0-9]{1,3}$")
    condition: Condition
    sus: list[int] = Field(min_length=10, max_length=10)
    tlx: list[int] = Field(min_length=6, max_length=6)

    def model_post_init(self, _context: object) -> None:
        if any(not 1 <= v <= 5 for v in self.sus):
            raise ValueError("SUS answers are 1-5")
        if any(not 0 <= v <= 100 for v in self.tlx):
            raise ValueError("NASA-TLX ratings are 0-100")


class StudyError(ValueError):
    """A study record that does not fit the plan."""


def connect_study(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def check_trial(plan: StudyPlan, trial: TrialIn) -> tuple[StudyClip, str | None]:
    """The trial's clip and, for an assisted trial, the participant's job for it.

    Refused unless the clip belongs to that participant's block in that condition.
    """
    if trial.participant not in plan.participants:
        raise StudyError(f"unknown participant {trial.participant}")
    condition, keys = plan.blocks(trial.participant)[trial.block - 1]
    if condition != trial.condition:
        raise StudyError(f"block {trial.block} of {trial.participant} is {condition}")
    if trial.clip not in keys:
        raise StudyError(f"clip {trial.clip} is not in block {trial.block} of {trial.participant}")
    if trial.condition == "manual":
        if trial.right is None or trial.left is None:
            raise StudyError("a manual trial needs both counts")
        return plan.clips[trial.clip], None
    job_id = plan.jobs.get(trial.participant, {}).get(trial.clip)
    if job_id is None:
        raise StudyError(f"no job for {trial.participant} on clip {trial.clip}")
    return plan.clips[trial.clip], job_id


def record_trial(
    conn: sqlite3.Connection,
    trial: TrialIn,
    clip: StudyClip,
    final: tuple[int, int],
    job_id: str | None,
    revision: int | None,
) -> None:
    with transaction(conn):
        if conn.execute(
            "SELECT 1 FROM trials WHERE participant = ? AND clip = ?",
            (trial.participant, trial.clip),
        ).fetchone():
            raise StudyError(f"{trial.participant} already finished clip {trial.clip}")
        conn.execute(
            "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trial.participant,
                trial.block,
                trial.condition,
                trial.clip,
                int(clip.role == "practice"),
                trial.started_at,
                trial.finished_at,
                trial.active_ms,
                trial.pauses,
                final[0],
                final[1],
                job_id,
                revision,
                trial.note,
            ),
        )


def record_questionnaire(conn: sqlite3.Connection, q: QuestionnaireIn, now: str) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT OR REPLACE INTO questionnaires VALUES (?, ?, ?, ?, ?)",
            (q.participant, q.condition, json.dumps(q.sus), json.dumps(q.tlx), now),
        )


def progress(conn: sqlite3.Connection, participant: str) -> dict[str, Any]:
    done = [
        r["clip"]
        for r in conn.execute("SELECT clip FROM trials WHERE participant = ?", (participant,))
    ]
    forms = [
        r["condition"]
        for r in conn.execute(
            "SELECT condition FROM questionnaires WHERE participant = ?", (participant,)
        )
    ]
    return {"participant": participant, "clips_done": done, "questionnaires_done": forms}


def export_csv(conn: sqlite3.Connection, table: Literal["trials", "questionnaires"]) -> str:
    if table not in ("trials", "questionnaires"):  # a fixed table name, never user input
        raise ValueError(f"unknown study table {table!r}")
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY participant").fetchall()
    out = io.StringIO()
    if rows:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(rows[0].keys())
        writer.writerows([tuple(r) for r in rows])
    return out.getvalue()
