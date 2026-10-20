"""Analyse the same-fish question re-test (docs/review_retest.md) from its running service.

For every reviewer in ``retest/plan.json`` who has made at least one decision, reads each
clip's latest revision, review events and changed tracks, and writes ``retest/results.json``:

- per clip: reference, automatic and reviewed counts, verdicts, changed tracks (described as
  in the reviewed sample), answers to the same-fish question, and for set B the missed
  passages the review made up;
- per reviewer, over complete clips (every track decided): count errors and nMAE for all
  clips, set A and set B, fragment counts, answers, recovered passages and timing;
- both targets per reviewer and over the reviewers (``passagewatch.evaluation.retest``);
- R1 and R2 on set A, from ``review_sample/results.json``, as historical context.

Example:
    scripts/retest_service.sh start
    uv run python scripts/analyze_retest.py --url http://127.0.0.1:8012
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from passagewatch.evaluation import retest
from passagewatch.evaluation.review_sample import clip_summary, review_timing
from passagewatch.service.client import changes, fetch, review_events

REPO_ROOT = Path(__file__).resolve().parents[1]


def totals(clips: list[dict[str, Any]]) -> dict[str, Any]:
    passages = sum(sum(c["reference"]) for c in clips)
    automatic = sum(c["automatic_error"] for c in clips)
    reviewed = sum(c["reviewed_error"] for c in clips)
    return {
        "clips": len(clips),
        "passages": passages,
        "automatic_error": automatic,
        "reviewed_error": reviewed,
        "nmae_automatic": automatic / passages if passages else None,
        "nmae_reviewed": reviewed / passages if passages else None,
        "fragment_counts": sum(c["fragment_counts"] for c in clips),
        "answers": {key: sum(c["answers"][key] for c in clips) for key in ("same", "different")},
    }


def historical(path: Path) -> dict[str, Any]:
    """R1's and R2's set-A figures from the reviewed sample (the page without the question)."""
    if not path.is_file():
        return {}
    sample = json.loads(path.read_text(encoding="utf-8"))["reviewers"]
    found = {}
    for reviewer, data in sample.items():
        clips = [data["clips"][code] for code in retest.SET_A if code in data["clips"]]
        found[reviewer] = {
            "clips": len(clips),
            "automatic_error": sum(c["automatic_error"] for c in clips),
            "reviewed_error": sum(c["reviewed_error"] for c in clips),
            "fragment_counts": sum(retest.fragment_counts(c.get("changes", [])) for c in clips),
        }
    return found


def analyze(client: httpx.Client, plan: dict[str, Any], sample_results: Path) -> dict[str, Any]:
    """The re-test's results from a service holding the plan's jobs."""
    reviewers: dict[str, Any] = {}
    not_started = []
    for reviewer in plan["reviewers"]:
        jobs = plan["jobs"][reviewer]
        events = {code: review_events(client, job_id) for code, job_id in jobs.items()}
        if not any(events.values()):
            not_started.append(reviewer)
            continue
        clips: dict[str, Any] = {}
        for code in sorted(jobs):
            clip = plan["clips"][code]
            labels, reviewed, added = fetch(client, jobs[code])
            summary = clip_summary(
                labels,
                reviewed,
                (clip["automatic"][0], clip["automatic"][1]),
                (clip["reference"][0], clip["reference"][1]),
                added,
            )
            summary["set"] = clip["set"]
            summary["sample_code"] = clip["sample_code"]
            summary["changes"] = changes(client, jobs[code], labels)
            summary["fragment_counts"] = retest.fragment_counts(summary["changes"])
            summary["answers"] = retest.answers(events[code])
            summary["complete"] = summary["verdicts"]["unreviewed"] == 0
            if clip["set"] == "B":
                shortfall, made_up = retest.recovered(
                    clip["automatic"], summary["reviewed"], clip["reference"]
                )
                summary["missed_passages"] = shortfall
                summary["recovered_passages"] = made_up
            clips[code] = summary
        complete = [c for c in clips.values() if c["complete"]]
        set_a = totals([c for c in complete if c["set"] == "A"])
        set_b_clips = [c for c in complete if c["set"] == "B"]
        every = totals(complete)
        reviewers[reviewer] = {
            "clips": clips,
            "incomplete_clips": sorted(k for k, c in clips.items() if not c["complete"]),
            "all": every,
            "set_a": set_a,
            "set_b": totals(set_b_clips)
            | {
                "missed_passages": sum(c["missed_passages"] for c in set_b_clips),
                "recovered_passages": sum(c["recovered_passages"] for c in set_b_clips),
            },
            "timing": review_timing(
                {code: [e["created_at"] for e in found] for code, found in events.items()}
            ),
            "target_no_harm": (
                every["reviewed_error"] <= every["automatic_error"] if complete else None
            ),
            "target_fragment_counts": (
                set_a["fragment_counts"] <= retest.MAX_FRAGMENT_COUNTS if set_a["clips"] else None
            ),
        }
    return {
        "design": plan["design"],
        "release": plan["release"],
        "review_page_commit": plan["review_page_commit"],
        "reviewers": reviewers,
        "not_started": not_started,
        "targets": {
            "no_harm": retest.group_target({r: d["target_no_harm"] for r, d in reviewers.items()}),
            "fragment_counts": retest.group_target(
                {r: d["target_fragment_counts"] for r, d in reviewers.items()}
            ),
        },
        "historical_set_a": historical(sample_results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8012")
    parser.add_argument("--plan", type=Path, default=REPO_ROOT / "retest/plan.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "retest/results.json")
    parser.add_argument(
        "--sample-results", type=Path, default=REPO_ROOT / "review_sample/results.json"
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.url, timeout=60) as client:
        result = analyze(client, plan, args.sample_results)
    reviewers = result["reviewers"]
    not_started = result["not_started"]
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for reviewer, r in reviewers.items():
        print(
            f"{reviewer}: {r['all']['clips']}/{len(r['clips'])} clips complete; error reviewed"
            f" {r['all']['reviewed_error']} vs automatic {r['all']['automatic_error']};"
            f" fragment counts on set A {r['set_a']['fragment_counts']}; answers"
            f" {r['all']['answers']}; set B recovered {r['set_b']['recovered_passages']}"
            f" of {r['set_b']['missed_passages']}"
        )
    for name, target in result["targets"].items():
        print(f"target {name}: {target['outcome']} ({target['met_by']} of {target['reviewers']})")
    if not_started:
        print(f"not started: {', '.join(not_started)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
