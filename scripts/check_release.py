"""Release gate: check that a release manifest allows the release to be published.

Run by ``.github/workflows/release.yml`` on a release tag, and before tagging. The trained
weights and data are not in git, so the counting evaluation and runtime parity are run
locally and recorded in the manifest (``make_release_manifest.py``); this gate checks that
record. It fails unless:

- ``releases/manifests/<release>.json`` exists and names that release;
- its code commit is in the history being released;
- it records at least one development evaluation report with an nMAE interval;
- every runtime parity check it records found 0 clips with different counts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from passagewatch.service.release import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", required=True)
    args = parser.parse_args(argv)

    path = REPO_ROOT / "releases/manifests" / f"{args.release}.json"
    if not path.is_file():
        print(f"FAIL no release manifest {path.relative_to(REPO_ROOT)}")
        return 1
    manifest = read_manifest(path)
    problems = []
    if manifest.release != args.release:
        problems.append(f"the manifest names {manifest.release}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", manifest.code_commit, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        problems.append(f"code commit {manifest.code_commit[:12]} is not in this history")
    reports = manifest.evaluation.get("reports", [])
    if not reports or any("nmae_95ci" not in r for r in reports):
        problems.append("no development evaluation with an nMAE interval is recorded")
    for check in manifest.evaluation.get("parity", []):
        if check["clips_with_different_counts"] != 0:
            differ = check["clips_with_different_counts"]
            problems.append(f"parity with {check['report']}: {differ} clips differ")
    for r in reports:
        print(f"  {r['partition']:8} {r['runtime']:12} nMAE {r['nmae']} {r['nmae_95ci']}")
    parity = manifest.evaluation.get("parity", [])
    print(f"  parity checks: {len(parity)}, image {manifest.image_digest}")
    for p in problems:
        print(f"FAIL {p}")
    print(f"{args.release}: {'release gate passed' if not problems else 'release gate FAILED'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
