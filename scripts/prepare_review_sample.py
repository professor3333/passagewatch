"""Select the reviewed development sample, analyse it with the release, write the plan.

Follows ``docs/review_sample.md`` (selection in ``passagewatch.evaluation.review_sample``):

1. Candidates are the kenai-holdout-v1 clips with the release's automatic counts from
   ``runs/neural/yolox-tiny-t1/report-holdout.json``, except the usability study's clips,
   the demo examples, and any clip named in the documentation.
2. Fifty clips are drawn, stratified by camera and reference passages, with seed 0, and
   given neutral codes ``s01``-``s50`` in a random order. Reviewer ``R2`` reviews the first
   fifteen codes.
3. Each clip is uploaded to a running service that serves the release and analysed; its
   automatic counts must equal the report's. Every reviewer gets their own job per clip (a
   result-cache hit), so nobody sees another reviewer's decisions.
4. ``review_sample/plan.json`` (committed before any review) and one link list per reviewer
   (``review_sample/R1.md``, ``R2.md``) are written. The reference counts are in the plan,
   which reviewers do not open. The plan records the excluded clips, so ``--verify`` can
   recompute the selection later, whatever the documentation names by then.

Example:
    scripts/review_sample_service.sh start
    uv run python scripts/prepare_review_sample.py --url http://127.0.0.1:8011
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from passagewatch.evaluation import review_sample, study_design
from passagewatch.ingestion.metadata import load_metadata
from passagewatch.service.client import create_job, upload, wait

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "runs/neural/yolox-tiny-t1/report-holdout.json"
METADATA = REPO_ROOT / "data/extracted/cfc/fish_counting_metadata/metadata/kenai-train.json"
FRAMES = REPO_ROOT / "data/extracted/cfc/kenai-holdout-v1/kenai-train"
OUT = REPO_ROOT / "review_sample"
RELEASE = "passagewatch-0.3.1"
REVIEWERS = ("R1", "R2")


def excluded_names() -> set[str]:
    """Clips already shown to people or named in the documentation."""
    study = json.loads((REPO_ROOT / "study/plan.json").read_text(encoding="utf-8"))
    demos = json.loads((REPO_ROOT / "demos/catalog.json").read_text(encoding="utf-8"))
    names = {c["clip_name"] for c in study["clips"].values()}
    return names | {d["source"]["clip_name"] for d in demos["demos"]}


def verify(plan_path: Path) -> int:
    """Check that the plan's clips and codes follow from its recorded exclusions."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    pool = study_design.candidates_from_report(report, load_metadata(METADATA).clips)
    excluded = set(plan["excluded"])
    chosen = review_sample.select([c for c in pool if c.clip_name not in excluded])
    expected = {f"s{i + 1:02d}": c.clip_name for i, c in enumerate(chosen)}
    actual = {key: clip["clip_name"] for key, clip in plan["clips"].items()}
    if expected != actual:
        print("the plan's selection does not follow from its exclusions", file=sys.stderr)
        return 1
    print(f"{plan_path}: the selection of {len(actual)} clips reproduces")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8011")
    parser.add_argument("--release", default=RELEASE)
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true", help="select and print, no uploads")
    parser.add_argument(
        "--verify", action="store_true", help="recompute the plan's selection from its exclusions"
    )
    args = parser.parse_args()

    plan_path = args.out / "plan.json"
    if args.verify:
        return verify(plan_path)
    if plan_path.exists() and not args.dry_run:
        print(f"{plan_path} exists; the sample is prepared once", file=sys.stderr)
        return 1
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metadata = load_metadata(METADATA).clips
    pool = study_design.candidates_from_report(report, metadata)
    texts = [(REPO_ROOT / "README.md").read_text(encoding="utf-8")] + [
        p.read_text(encoding="utf-8") for p in sorted((REPO_ROOT / "docs").rglob("*.md"))
    ]
    documented = study_design.documented_names(texts, [c.clip_name for c in pool])
    skip = excluded_names() | documented
    eligible = [c for c in pool if c.clip_name not in skip]
    chosen = review_sample.select(eligible)
    codes = {f"s{i + 1:02d}": c for i, c in enumerate(chosen)}
    second = sorted(codes)[: review_sample.SECOND_REVIEWER_CLIPS]
    assignments = {"R1": sorted(codes), "R2": second}

    print(f"{len(pool)} candidates, {len(skip & {c.clip_name for c in pool})} excluded")
    print(f"{len(eligible)} eligible; {len(chosen)} chosen; R2 reviews {', '.join(second)}")
    strata: dict[str, int] = {}
    for c in chosen:
        key = f"{c.camera} {review_sample.PASSAGE_BINS[review_sample.stratum(c)[1]]}"
        strata[key] = strata.get(key, 0) + 1
    for key, n in sorted(strata.items()):
        print(f"  {key}: {n}")
    print(f"  {sum(c.seconds for c in chosen) / 60:.0f} minutes of recordings in all")
    if args.dry_run:
        return 0

    with httpx.Client(base_url=args.url, timeout=120) as client:
        version = client.get("/v1/model-info").json()["pipeline_version"]
        if version != args.release:
            print(f"the service runs {version}, not {args.release}", file=sys.stderr)
            return 1
        clips: dict[str, dict[str, Any]] = {}
        for key, c in codes.items():
            clip_id = upload(client, FRAMES / c.clip_name, metadata[c.clip_name])
            results = wait(client, create_job(client, clip_id, f"sample-{key}"), args.job_timeout)
            automatic = (results["automatic"]["right"], results["automatic"]["left"])
            if automatic != c.automatic:
                print(f"{key}: service counts {automatic}, report {c.automatic}", file=sys.stderr)
                return 1
            clips[key] = {
                "clip_name": c.clip_name,
                "clip_id": clip_id,
                "camera": c.camera,
                "num_frames": c.num_frames,
                "framerate": c.framerate,
                "reference": list(c.reference),
                "automatic": list(automatic),
            }
            print(f"  {key} analysed: automatic {automatic}")
        jobs: dict[str, dict[str, str]] = {}
        for reviewer, keys in assignments.items():
            jobs[reviewer] = {}
            for key in keys:
                job_id = create_job(client, clips[key]["clip_id"], f"sample-{reviewer}-{key}")
                wait(client, job_id, args.job_timeout)
                jobs[reviewer][key] = job_id

    plan = {
        "design": "docs/review_sample.md",
        "release": args.release,
        "seed": review_sample.SEED,
        "excluded": sorted(skip & {c.clip_name for c in pool}),
        "clips": clips,
        "assignments": assignments,
        "jobs": jobs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for reviewer, keys in assignments.items():
        lines = [
            f"# Review sample: reviewer {reviewer}",
            "",
            "Follow the instructions in docs/review_sample.md. Open each link, review every "
            "track, and tick the clip off here when you are done.",
            "",
        ]
        lines += [f"- [ ] {key}: {args.url}/#/jobs/{jobs[reviewer][key]}" for key in keys]
        (args.out / f"{reviewer}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {plan_path} and the reviewers' link lists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
