"""Analyse the reviewed development sample (docs/review_sample.md) from the running service.

For every reviewer's job in ``review_sample/plan.json``, reads the latest revision's tracks,
the reviewed counts and the passages the reviewer added, and writes
``review_sample/results.json``:

- per clip and reviewer: CFC reference, automatic and reviewed counts, each track's verdict,
  and added passages; reviewed counts that differ from the reference are listed as
  disagreements, not resolved;
- per reviewer: nMAE of the automatic and the reviewed counts against the reference, and how
  often the reviewer changed a track's decision in each triage state;
- between R1 and R2, on the clips both reviewed: the share of tracks with the same final
  outcome, Cohen's kappa, and every disagreement;
- per changed track: where it lies relative to the counting line and which counted tracks
  are next to it in time (descriptive, not a judgment);
- per reviewer: when the decisions were recorded, per clip and overall.

A clip with unreviewed tracks is reported as incomplete. When the plan records amendments,
only the clips assigned by the latest one are analysed.

Example:
    scripts/review_sample_service.sh start
    uv run python scripts/analyze_review_sample.py --url http://127.0.0.1:8011
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from passagewatch.evaluation.review_sample import (
    TrackLabel,
    agreement,
    clip_summary,
    describe_changes,
    review_timing,
    track_labels,
    track_paths,
    triage_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = 200
OBSERVATION_FRAMES = 500  # the API's limit per request


def fetch(client: httpx.Client, job_id: str) -> tuple[list[TrackLabel], tuple[int, int], int]:
    tracks: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            client.get(f"/v1/jobs/{job_id}/tracks", params={"offset": offset, "limit": PAGE})
            .raise_for_status()
            .json()
        )
        tracks += page["tracks"]
        offset += PAGE
        if offset >= page["total"]:
            break
    results = client.get(f"/v1/jobs/{job_id}/results").raise_for_status().json()
    counts = results["reviewed"] or results["automatic"]
    report = client.get(f"/v1/jobs/{job_id}/export", params={"format": "json"}).json()
    added = sum(p["state"] == "added" for p in report.get("added_passages", []))
    return track_labels(tracks), (counts["right"], counts["left"]), added


def changes(client: httpx.Client, job_id: str, labels: list[TrackLabel]) -> list[dict[str, Any]]:
    """The changed tracks of a job, described from its observations."""
    if not any(label.verdict in ("removed", "counted", "redirected") for label in labels):
        return []
    job = client.get(f"/v1/jobs/{job_id}").raise_for_status().json()
    clip = client.get(f"/v1/clips/{job['clip_id']}").raise_for_status().json()
    boxes: list[dict[str, Any]] = []
    for start in range(0, clip["num_frames"], OBSERVATION_FRAMES):
        stop = min(clip["num_frames"], start + OBSERVATION_FRAMES)
        params = {"start": start, "stop": stop}
        page = client.get(f"/v1/jobs/{job_id}/observations", params=params)
        boxes += page.raise_for_status().json()["boxes"]
    paths = track_paths(boxes, clip["width"])
    return describe_changes(labels, paths, job["counting"]["line_x_normalized"])


def event_times(client: httpx.Client, job_id: str) -> list[str]:
    events = client.get(f"/v1/jobs/{job_id}/reviews").raise_for_status().json()["events"]
    return [e["created_at"] for e in events]


def nmae(clips: list[dict[str, Any]], key: str) -> float | None:
    passages = sum(sum(c["reference"]) for c in clips)
    return sum(c[key] for c in clips) / passages if passages else None


def assignments(plan: dict[str, Any]) -> dict[str, list[str]]:
    """Each reviewer's clips: the latest amendment's assignment, or the original one."""
    amendments = plan.get("amendments") or []
    chosen = amendments[-1]["assignments"] if amendments else plan["assignments"]
    return {reviewer: list(keys) for reviewer, keys in chosen.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--plan", type=Path, default=REPO_ROOT / "review_sample/plan.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "review_sample/results.json")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    labels: dict[str, dict[str, list[TrackLabel]]] = {}
    reviewers: dict[str, Any] = {}
    with httpx.Client(base_url=args.url, timeout=60) as client:
        for reviewer, keys in assignments(plan).items():
            labels[reviewer] = {}
            clips: dict[str, Any] = {}
            times: dict[str, list[str]] = {}
            for key in sorted(keys):
                job_id = plan["jobs"][reviewer][key]
                clip = plan["clips"][key]
                found, reviewed, added = fetch(client, job_id)
                labels[reviewer][key] = found
                clips[key] = clip_summary(
                    found,
                    reviewed,
                    (clip["automatic"][0], clip["automatic"][1]),
                    (clip["reference"][0], clip["reference"][1]),
                    added,
                )
                clips[key]["changes"] = changes(client, job_id, found)
                times[key] = event_times(client, job_id)
            complete = [c for c in clips.values() if c["verdicts"]["unreviewed"] == 0]
            every = [label for found in labels[reviewer].values() for label in found]
            reviewers[reviewer] = {
                "clips": clips,
                "complete_clips": len(complete),
                "assigned_clips": len(clips),
                "nmae_automatic": nmae(complete, "automatic_error"),
                "nmae_reviewed": nmae(complete, "reviewed_error"),
                "triage": triage_table(every),
                "timing": review_timing(times),
                "recording_minutes": round(
                    sum(
                        plan["clips"][k]["num_frames"] / plan["clips"][k]["framerate"]
                        for k in clips
                    )
                    / 60,
                    1,
                ),
                "disagreements_with_reference": [
                    {"clip": key, "reference": c["reference"], "reviewed": c["reviewed"]}
                    for key, c in clips.items()
                    if not c["agrees_with_reference"] and c["verdicts"]["unreviewed"] == 0
                ],
            }
    between: dict[str, Any] = {}
    if {"R1", "R2"} <= set(labels):
        between = agreement(labels["R1"], labels["R2"])
        shared = between["clips"]
        # Counts are compared only on clips both reviewers completed.
        complete = [
            k
            for k in shared
            if reviewers["R1"]["clips"][k]["verdicts"]["unreviewed"] == 0
            and reviewers["R2"]["clips"][k]["verdicts"]["unreviewed"] == 0
        ]
        between["clips_complete_for_both"] = complete
        between["clips_with_same_counts"] = sum(
            reviewers["R1"]["clips"][k]["reviewed"] == reviewers["R2"]["clips"][k]["reviewed"]
            for k in complete
        )

    result = {"design": plan["design"], "release": plan["release"], "reviewers": reviewers}
    result["agreement_R1_R2"] = between
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for reviewer, r in reviewers.items():
        print(
            f"{reviewer}: {r['complete_clips']}/{r['assigned_clips']} clips complete; "
            f"nMAE automatic {r['nmae_automatic']}, reviewed {r['nmae_reviewed']}; "
            f"{len(r['disagreements_with_reference'])} clips differ from the reference"
        )
    if between:
        print(
            f"R1 vs R2: {between['same_outcome']}/{between['tracks']} tracks agree, "
            f"kappa {between['kappa']:.2f}; {between['clips_with_same_counts']}/"
            f"{len(between['clips_complete_for_both'])} clips complete for both share counts"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
