"""Select the usability study's clips, analyse them with the release, and write the plan.

Follows ``docs/usability_study.md`` (selection rules in
``passagewatch.evaluation.study_design``):

1. Candidates are the kenai-holdout-v1 clips with the release's automatic counts from
   ``runs/neural/yolox-tiny-t1/report-holdout.json`` (its selected epoch and threshold).
   Clips named anywhere in ``README.md`` or ``docs/`` are not eligible.
2. Practice clips and the two matched sets are selected with seed 0.
3. Each clip's frames are zipped, uploaded to a running service that serves the release, and
   analysed. Every participant then gets their own job for each clip they review (a
   result-cache hit), so nobody sees another participant's corrections.
4. The service's automatic counts must equal the report's (the ONNX runtime counts like
   the evaluated one); any difference stops the script.
5. ``study/plan.json`` is written. It is committed before the first session.

Run the service for the study with a long upload retention, for example
``PASSAGEWATCH_UPLOAD_RETENTION_HOURS=2160`` (90 days), and the release active.

Example:
    uv run python scripts/prepare_usability_study.py --url http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from passagewatch.evaluation import study_design
from passagewatch.ingestion.metadata import load_metadata
from passagewatch.service.client import create_job, upload, wait

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "runs/neural/yolox-tiny-t1/report-holdout.json"
METADATA = REPO_ROOT / "data/extracted/cfc/fish_counting_metadata/metadata/kenai-train.json"
FRAMES = REPO_ROOT / "data/extracted/cfc/kenai-holdout-v1/kenai-train"
PLAN = REPO_ROOT / "study/plan.json"
RELEASE = "passagewatch-0.3.0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--participants", default="D1,P1,P2,P3,P4")
    parser.add_argument("--release", default=RELEASE)
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--plan", type=Path, default=PLAN, help="where to write the plan")
    parser.add_argument("--dry-run", action="store_true", help="select and print, no uploads")
    args = parser.parse_args()

    if args.plan.exists() and not args.dry_run:
        print(f"{args.plan} exists; the study is prepared once", file=sys.stderr)
        return 1
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metadata = load_metadata(METADATA).clips
    pool = study_design.candidates_from_report(report, metadata)
    texts = [(REPO_ROOT / "README.md").read_text(encoding="utf-8")] + [
        p.read_text(encoding="utf-8") for p in sorted((REPO_ROOT / "docs").rglob("*.md"))
    ]
    documented = study_design.documented_names(texts, [c.clip_name for c in pool])
    eligible = study_design.eligible(pool, documented)
    practice, s1, s2 = study_design.select(eligible)
    clips, practice_keys, sets = study_design.assign_codes(practice, s1, s2)
    participants = args.participants.split(",")
    blocks = study_design.latin_square(participants)

    print(f"{len(pool)} candidates, {len(documented)} documented, {len(eligible)} eligible")
    for name, members in (("S1", s1), ("S2", s2)):
        print(
            f"{name}: {study_design.cameras(members)}, "
            f"{sum(c.passages for c in members)} passages, "
            f"{sum(c.seconds for c in members):.0f} s, "
            f"automatic error {sum(c.automatic_error for c in members)}"
        )
    for key, c in sorted(clips.items()):
        print(f"  {key} {c.clip_name} {c.camera} {c.num_frames} frames {c.seconds:.0f} s")
    if args.dry_run:
        return 0

    role = {c.clip_name: "practice" for c in practice}
    role |= {c.clip_name: "S1" for c in s1} | {c.clip_name: "S2" for c in s2}
    with httpx.Client(base_url=args.url, timeout=120) as client:
        version = client.get("/v1/model-info").json()["pipeline_version"]
        if version != args.release:
            print(f"the service runs {version}, not {args.release}", file=sys.stderr)
            return 1
        entries: dict[str, dict[str, Any]] = {}
        for key, c in sorted(clips.items()):
            clip_id = upload(client, FRAMES / c.clip_name, metadata[c.clip_name])
            results = wait(client, create_job(client, clip_id, f"study-{key}"), args.job_timeout)
            automatic = (results["automatic"]["right"], results["automatic"]["left"])
            if automatic != c.automatic:
                print(f"{key}: service counts {automatic}, report {c.automatic}", file=sys.stderr)
                return 1
            entries[key] = {
                "clip_name": c.clip_name,
                "clip_id": clip_id,
                "role": role[c.clip_name],
                "camera": c.camera,
                "num_frames": c.num_frames,
                "framerate": c.framerate,
                "reference": list(c.reference),
                "automatic": list(automatic),
            }
            print(f"  {key} analysed: automatic {automatic}")
        jobs: dict[str, dict[str, str]] = {}
        for participant, participant_blocks in blocks.items():
            jobs[participant] = {}
            for index, block in enumerate(participant_blocks):
                if block["condition"] != "assisted":
                    continue
                for key in [practice_keys[index], *sets[block["set"]]]:
                    job_id = create_job(
                        client, entries[key]["clip_id"], f"study-{participant}-{key}"
                    )
                    wait(client, job_id, args.job_timeout)
                    jobs[participant][key] = job_id

    plan = {
        "design": "docs/usability_study.md",
        "release": args.release,
        "seed": study_design.SEED,
        "clips": entries,
        "practice": practice_keys,
        "sets": sets,
        "participants": blocks,
        "jobs": jobs,
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.plan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
