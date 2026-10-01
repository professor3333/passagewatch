"""Validate CFC clips, then write the versioned clip manifest that records each partition.

Writes the validation report and ``data/manifests/splits/cfc/<version>.{parquet,json}``.
An existing version is never changed: rerunning with identical results is a no-op, and
any difference requires a new version name.

Examples:
    uv run python scripts/build_manifest.py --version tiny-v1
    uv run python scripts/build_manifest.py --version full-v1      # annotations only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from passagewatch.ingestion.cfc import CfcLayout
from passagewatch.ingestion.inventory import sha256_of
from passagewatch.ingestion.manifest import (
    ManifestVersionError,
    build_rows,
    manifest_paths,
    write_manifest,
)
from passagewatch.validation.cfc import validate_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORIES = {
    "tiny": ("tiny_dataset", "fish_counting_annotations", "fish_counting_metadata"),
    "full": ("fish_counting_annotations", "fish_counting_metadata"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="e.g. tiny-v1 or full-v1")
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "data/manifests")
    args = parser.parse_args(argv)

    out_dir = args.manifest_dir / "splits/cfc"
    try:
        manifest_paths(out_dir, args.version)
    except ValueError as exc:
        parser.error(str(exc))
    subset = args.version.split("-")[0]
    layout = (
        CfcLayout.tiny(args.extract_dir) if subset == "tiny" else CfcLayout.full(args.extract_dir)
    )

    report = validate_dataset(layout)
    report_path = args.manifest_dir / f"validation/cfc/{subset}.json"
    report.write_json(report_path)

    inputs = {"validation_report": sha256_of(report_path)}
    for name in INVENTORIES[subset]:
        inventory = args.manifest_dir / f"inventory/cfc/{name}.parquet"
        inputs[f"inventory/{name}"] = sha256_of(inventory)

    rows = build_rows(layout, report)
    try:
        paths = write_manifest(rows, out_dir, args.version, inputs=inputs)
    except ManifestVersionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    usable = sum(r["usable"] for r in rows)
    print(f"{len(rows)} clips ({usable} usable) -> {paths.parquet}")
    for partition in ("train", "val", "test"):
        selected = [r for r in rows if r["partition"] == partition]
        if selected:
            tuning = selected[0]["tuning_allowed"]
            print(f"  {partition:5} {len(selected):5d} clips  tuning_allowed={tuning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
