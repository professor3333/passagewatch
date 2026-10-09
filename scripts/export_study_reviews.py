"""Export the usability study's review actions, with each track's automatic outcome.

For every assisted trial in ``study/trials.csv``, reads the job's review events and its
tracks from the running study service, and keeps the events up to the revision the
participant's counts came from. Each row records the action and, for track actions, what
the release had decided about that track (outcome, direction, triage), so the analysis can
tell a corrected passage from a track that was turned into one.

Example:
    scripts/study_service.sh start
    uv run python scripts/export_study_reviews.py --url http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY = REPO_ROOT / "study"
PAGE = 200
FIELDS = [
    "participant",
    "clip",
    "practice",
    "job_id",
    "event_id",
    "action",
    "track_id",
    "direction",
    "frame_index",
    "automatic_outcome",
    "automatic_direction",
    "automatic_triage",
]


def automatic_tracks(client: httpx.Client, job_id: str) -> dict[int, dict[str, Any]]:
    tracks: dict[int, dict[str, Any]] = {}
    offset = 0
    while True:
        page = (
            client.get(f"/v1/jobs/{job_id}/tracks", params={"offset": offset, "limit": PAGE})
            .raise_for_status()
            .json()
        )
        tracks |= {t["track_id"]: t for t in page["tracks"]}
        offset += PAGE
        if offset >= page["total"]:
            return tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--trials", type=Path, default=STUDY / "trials.csv")
    parser.add_argument("--out", type=Path, default=STUDY / "review_actions.csv")
    args = parser.parse_args()

    with args.trials.open(encoding="utf-8") as fh:
        trials = [r for r in csv.DictReader(fh) if r["condition"] == "assisted"]
    rows = []
    with httpx.Client(base_url=args.url, timeout=60) as client:
        for trial in trials:
            job_id, revision = trial["job_id"], int(trial["revision"])
            tracks = automatic_tracks(client, job_id)
            history = client.get(f"/v1/jobs/{job_id}/reviews").raise_for_status().json()
            for event in history["events"]:
                if event["resulting_revision"] > revision:
                    continue
                track = tracks.get(event.get("track_id", -1), {})
                rows.append(
                    {
                        "participant": trial["participant"],
                        "clip": trial["clip"],
                        "practice": trial["practice"],
                        "job_id": job_id,
                        "event_id": event["event_id"],
                        "action": event["action"],
                        "track_id": event.get("track_id", ""),
                        "direction": event.get("direction", ""),
                        "frame_index": event.get("frame_index", ""),
                        "automatic_outcome": track.get("outcome", ""),
                        "automatic_direction": track.get("direction") or "",
                        "automatic_triage": track.get("triage") or "",
                    }
                )
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    by_action: dict[str, int] = {}
    for row in rows:
        by_action[row["action"]] = by_action.get(row["action"], 0) + 1
    print(f"wrote {len(rows)} review actions from {len(trials)} assisted trials to {args.out}")
    print(json.dumps(by_action, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
