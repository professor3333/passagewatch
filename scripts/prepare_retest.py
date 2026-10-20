"""Prepare the same-fish question re-test (docs/review_retest.md): clips, jobs, link lists.

1. The clips are set A (``s01``-``s15``) and set B (the first six undercounted clips of
   ``s16``-``s50``) of the reviewed sample (``review_sample/plan.json``), shown under new
   codes ``t01``-``t21`` in one order drawn with seed 0
   (``passagewatch.evaluation.retest``).
2. Each clip is uploaded to the re-test's service, which runs the release, and analysed. Its
   automatic counts must equal those recorded in the reviewed sample (the same pipeline).
3. Every reviewer ``T1``-``T4`` gets their own job per clip (a result-cache hit), so nobody
   sees another reviewer's decisions.
4. ``retest/plan.json`` records the clips, codes, sets, reference and automatic counts, the
   release and the commit whose review page is served. One link list per reviewer
   (``retest/T1.md``...) carries the instructions: those R2 had, with the new codes. Neither
   names the sets or what the re-test is about. Run once, from a clean checkout.

Example:
    scripts/retest_service.sh start
    uv run python scripts/prepare_retest.py --url http://127.0.0.1:8012
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from passagewatch.evaluation import retest
from passagewatch.ingestion.metadata import load_metadata
from passagewatch.service.client import create_job, upload, wait

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PLAN = REPO_ROOT / "review_sample/plan.json"
METADATA = REPO_ROOT / "data/extracted/cfc/fish_counting_metadata/metadata/kenai-train.json"
FRAMES = REPO_ROOT / "data/extracted/cfc/kenai-holdout-v1/kenai-train"
OUT = REPO_ROOT / "retest"
RELEASE = "passagewatch-0.3.3"

INSTRUCTIONS = """\
Thank you for reviewing. You check fish counts that a program suggests for short sonar
recordings, and correct them where they are wrong. Work alone, and take breaks whenever you
like; nothing is timed. It takes about an hour, and you can stop and continue later.

For each clip below, in order:

1. Open the link and play the recording. Use the speed control and the frame steps as needed.
2. Give **every track** in the list a decision. A clip is not complete until no track shows
   "not reviewed".
   - **Accept** a counted passage that is right, fish and direction; **Keep uncounted** an
     uncounted track that really did not pass.
   - **Not a fish: don't count** for a counted track that is not a fish, or that duplicates
     another track of the same fish.
   - **Change to** the other direction for a passage counted the wrong way.
   - **Count as passage → / ←** for an uncounted track whose fish did cross the line and
     stay across, and is not already counted by another track.
   - **Unresolved** when you cannot tell. This is a valid answer; do not guess.
3. Add every fish that crossed but has no track: **Add missed fish here** at the frame where
   it crosses. Check the **random audit** windows and mark them checked.
4. Tick the clip off in this list.

A fish that crosses and comes back does not count. Directions are as seen in the image
(→ right, ← left).

**One passage per fish.** The tracker sometimes splits one fish into several tracks, for
example one piece before the line and another after it, or puts two boxes on the same fish.
Such a fish counts **once**:

- If one of its tracks is already counted, keep its other tracks uncounted (**Keep
  uncounted**), or remove a second counted one (**Not a fish: don't count**).
- If none of its tracks is counted but the fish did cross and stay across, count it on
  **one** track (**Count as passage**) and keep the others uncounted.
- Before counting an uncounted track, look for another track on the same fish at the same
  time or just before and after it.
"""


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8012")
    parser.add_argument("--release", default=RELEASE)
    parser.add_argument("--job-timeout", type=float, default=1800.0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the clips, no uploads")
    args = parser.parse_args()

    plan_path = args.out / "plan.json"
    if plan_path.exists() and not args.dry_run:
        print(f"{plan_path} exists; the re-test is prepared once", file=sys.stderr)
        return 1
    sample = json.loads(SAMPLE_PLAN.read_text(encoding="utf-8"))["clips"]
    sets = {code: "A" for code in retest.SET_A} | {code: "B" for code in retest.set_b(sample)}
    order = retest.presentation_order(list(sets))
    for new, old in order.items():
        clip = sample[old]
        print(
            f"{new} = {old} (set {sets[old]}): reference {clip['reference']},"
            f" automatic {clip['automatic']}"
        )
    seconds = sum(sample[old]["num_frames"] / sample[old]["framerate"] for old in sets)
    print(f"{len(order)} clips, {seconds / 60:.1f} minutes of recordings")
    if args.dry_run:
        return 0
    if git("status", "--porcelain", "--untracked-files=no"):
        print(
            "uncommitted changes: the served review page must be a recorded commit", file=sys.stderr
        )
        return 1
    commit = git("rev-parse", "HEAD")

    metadata = load_metadata(METADATA).clips
    with httpx.Client(base_url=args.url, timeout=120) as client:
        version = client.get("/v1/model-info").json()["pipeline_version"]
        if version != args.release:
            print(f"the service runs {version}, not {args.release}", file=sys.stderr)
            return 1
        clips: dict[str, dict[str, Any]] = {}
        for new, old in order.items():
            source = sample[old]
            name = source["clip_name"]
            clip_id = upload(client, FRAMES / name, metadata[name])
            results = wait(client, create_job(client, clip_id, f"retest-{new}"), args.job_timeout)
            automatic = [results["automatic"]["right"], results["automatic"]["left"]]
            if automatic != source["automatic"]:
                print(
                    f"{new}: service counts {automatic}, reviewed sample {source['automatic']}",
                    file=sys.stderr,
                )
                return 1
            clips[new] = {
                "sample_code": old,
                "set": sets[old],
                "clip_name": name,
                "clip_id": clip_id,
                "camera": source["camera"],
                "num_frames": source["num_frames"],
                "framerate": source["framerate"],
                "reference": source["reference"],
                "automatic": automatic,
            }
            print(f"  {new} analysed: automatic {automatic}")
        jobs: dict[str, dict[str, str]] = {}
        for reviewer in retest.REVIEWERS:
            jobs[reviewer] = {}
            for new in order:
                job_id = create_job(client, clips[new]["clip_id"], f"retest-{reviewer}-{new}")
                wait(client, job_id, args.job_timeout)
                jobs[reviewer][new] = job_id

    plan = {
        "design": "docs/review_retest.md",
        "release": args.release,
        "review_page_commit": commit,
        "seed": retest.SEED,
        "clips": clips,
        "reviewers": list(retest.REVIEWERS),
        "jobs": jobs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    for reviewer in retest.REVIEWERS:
        lines = [f"# Reviewing fish counts: reviewer {reviewer}", "", INSTRUCTIONS, "## Clips", ""]
        lines += [f"- [ ] {new}: {args.url}/#/jobs/{jobs[reviewer][new]}" for new in order]
        (args.out / f"{reviewer}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {plan_path} and the reviewers' link lists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
