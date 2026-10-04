"""Record the published image's digest in a release manifest (once).

The release workflow publishes the image and its digest in the GitHub Release; this writes
the digest into ``releases/manifests/<release>.json``, which is then committed by the
project owner. A manifest's digest can be filled in once and never changed.

Example:
    uv run python scripts/record_image_digest.py --release passagewatch-0.3.0 \\
        --digest sha256:0123...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from passagewatch.service.release import read_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", required=True)
    parser.add_argument("--digest", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.digest):
        parser.error("the digest must look like sha256:<64 hex characters>")
    path = REPO_ROOT / "releases/manifests" / f"{args.release}.json"
    manifest = read_manifest(path)
    write_manifest(manifest.model_copy(update={"image_digest": args.digest}), path)
    print(f"{args.release}: image {args.digest} -> {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
