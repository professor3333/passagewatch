"""Replay recorded reviews: how many "count" decisions would have met the review page's
same-fish question (passagewatch.evaluation.same_fish), and how often the question appears.

Reads a review service's data directory directly (``service.db`` and the track artifacts).
The usability study, the reviewed sample and the documentation check each have one; none is
committed.

Examples:
    uv run python scripts/same_fish_coverage.py var/study \
        --jobs-from-study study/review_actions.csv
    uv run python scripts/same_fish_coverage.py var/review-sample \
        --jobs-from-sample review_sample/plan.json
    uv run python scripts/same_fish_coverage.py <data dir> --job job_28a64b41a2554cb9
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

from passagewatch.evaluation.same_fish import question_rate, replay
from passagewatch.service.artifacts import read_all_tracks


def study_jobs(path: Path) -> dict[str, list[str]]:
    """Assisted, non-practice jobs of the independent participants (not the developer)."""
    jobs: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["participant"].startswith("D") or row["practice"] != "0":
                continue
            jobs.setdefault(row["participant"], set()).add(row["job_id"])
    return {who: sorted(ids) for who, ids in sorted(jobs.items())}


def sample_jobs(path: Path) -> dict[str, list[str]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    amendments = plan.get("amendments") or []
    assignments = amendments[-1]["assignments"] if amendments else plan["assignments"]
    return {who: [plan["jobs"][who][key] for key in keys] for who, keys in assignments.items()}


def job_summary(conn: sqlite3.Connection, data_dir: Path, job_id: str) -> dict[str, int]:
    revisions = {
        int(revision): (artifact, json.loads(decisions))
        for revision, artifact, decisions in conn.execute(
            "SELECT revision, tracks_artifact, decisions_json FROM result_revisions"
            " WHERE job_id = ?",
            (job_id,),
        )
    }
    tracks = read_all_tracks(data_dir / "artifacts" / revisions[0][0])
    events = [
        (int(base), json.loads(payload))
        for base, payload in conn.execute(
            "SELECT base_revision, payload_json FROM review_events WHERE job_id = ?"
            " ORDER BY event_id",
            (job_id,),
        )
    ]
    found = replay(tracks, events, {r: d for r, (_, d) in revisions.items()})
    uncounted, with_question = question_rate(tracks)
    return {
        "count_decisions": len(found),
        "asked": sum(d.asked for d in found),
        "uncounted_tracks": uncounted,
        "with_question": with_question,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data_dir", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jobs-from-study", type=Path)
    group.add_argument("--jobs-from-sample", type=Path)
    group.add_argument("--job", action="append")
    args = parser.parse_args()
    if args.jobs_from_study:
        groups = study_jobs(args.jobs_from_study)
    elif args.jobs_from_sample:
        groups = sample_jobs(args.jobs_from_sample)
    else:
        groups = {"jobs": args.job}
    conn = sqlite3.connect(f"file:{args.data_dir / 'service.db'}?mode=ro", uri=True)
    totals = dict.fromkeys(("count_decisions", "asked", "uncounted_tracks", "with_question"), 0)
    for name, job_ids in groups.items():
        group_total = dict.fromkeys(totals, 0)
        for job_id in job_ids:
            for key, value in job_summary(conn, args.data_dir, job_id).items():
                group_total[key] += value
        print(
            f"{name}: {group_total['asked']} of {group_total['count_decisions']} count decisions"
            f" would have met the question; it appears on {group_total['with_question']} of"
            f" {group_total['uncounted_tracks']} uncounted tracks"
        )
        for key in totals:
            totals[key] += group_total[key]
    print(json.dumps(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
