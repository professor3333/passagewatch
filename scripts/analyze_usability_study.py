"""Analyse the usability study (docs/usability_study.md) from its exported records.

Reads the plan (reference and automatic counts) and the two CSV tables the study service
exports, then applies the declared analysis (``passagewatch.evaluation.usability``).
A clip interrupted by a technical fault is excluded with ``--exclude P1:c04:"reason"``.

Example:
    curl -s http://127.0.0.1:8010/v1/study/export/trials.csv > study/trials.csv
    curl -s http://127.0.0.1:8010/v1/study/export/questionnaires.csv > study/questionnaires.csv
    uv run python scripts/analyze_usability_study.py --out study/results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from passagewatch.evaluation.usability import analyze, trials_from_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY = REPO_ROOT / "study"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", type=Path, default=STUDY / "plan.json")
    parser.add_argument("--trials", type=Path, default=STUDY / "trials.csv")
    parser.add_argument("--questionnaires", type=Path, default=STUDY / "questionnaires.csv")
    parser.add_argument("--exclude", action="append", default=[], help="PARTICIPANT:CLIP:REASON")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    exclusions = []
    for item in args.exclude:
        participant, clip, reason = item.split(":", 2)
        exclusions.append({"participant": participant, "clip": clip, "reason": reason})
    excluded = {(e["participant"], e["clip"]) for e in exclusions}
    with args.trials.open(encoding="utf-8") as fh:
        trials = trials_from_rows(list(csv.DictReader(fh)), plan, excluded)
    forms = []
    if args.questionnaires.is_file():
        with args.questionnaires.open(encoding="utf-8") as fh:
            forms = [
                r | {"sus": json.loads(r["sus"]), "tlx": json.loads(r["tlx"])}
                for r in csv.DictReader(fh)
            ]

    result = analyze(trials, forms) | {"release": plan["release"], "exclusions": exclusions}
    for person, r in result["participants"].items():
        print(
            f"{person} ({r['role']}): time saving {r['time_saving']:.0%} "
            f"[{r['time_saving_ci'][0]:.0%}, {r['time_saving_ci'][1]:.0%}], "
            f"nMAE manual {r['nmae']['manual']:.3f} vs assisted {r['nmae']['assisted']:.3f} "
            f"(difference {r['nmae_difference']:+.3f} [{r['nmae_difference_ci'][0]:+.3f}, "
            f"{r['nmae_difference_ci'][1]:+.3f}]), target met: {r['target_met']}"
        )
    group = result["group"]
    if group is None:
        print("fewer than 3 independent participants: per-person (pilot) results only")
    else:
        print(
            f"independent group {', '.join(group['participants'])}: "
            f"time saving {group['time_saving']:.0%} "
            f"[{group['time_saving_ci'][0]:.0%}, {group['time_saving_ci'][1]:.0%}], "
            f"nMAE difference {group['nmae_difference']:+.3f}, target met: {group['target_met']}"
        )
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
