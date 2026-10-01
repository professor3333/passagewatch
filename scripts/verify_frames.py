"""Check that every file under a frame directory matches a committed SHA-256 inventory.

Used on GPU hosts after streaming a subset: the frames there must be byte-identical to the
frames that were validated and recorded in the manifest. Files missing from the directory
are allowed (a host may hold only part of a subset); unknown or changed files are errors.

Example:
    python scripts/verify_frames.py --root /tmp/cfc/kenai-dev-v1-train \\
        --inventory data/manifests/inventory/cfc/kenai-dev-v1.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from passagewatch.ingestion.inventory import build_inventory, read_inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args(argv)

    expected = {e.path: e.sha256 for e in read_inventory(args.inventory)}
    actual = build_inventory(args.root)
    unknown = [e.path for e in actual if e.path not in expected]
    changed = [e.path for e in actual if e.path in expected and expected[e.path] != e.sha256]
    print(f"{len(actual)} files checked against {len(expected)} inventoried")
    for label, paths in (("unknown", unknown), ("changed", changed)):
        if paths:
            print(f"{len(paths)} {label} files, e.g. {paths[:5]}", file=sys.stderr)
    return 1 if unknown or changed else 0


if __name__ == "__main__":
    sys.exit(main())
