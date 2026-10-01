"""Validate CFC clips, then write the versioned clip manifest that records each partition.

Writes the validation report and ``data/manifests/splits/cfc/<version>.{parquet,json}``.
An existing version is never changed: rerunning with identical results is a no-op, and
any difference requires a new version name.

Examples:
    uv run python scripts/build_manifest.py --version tiny-v1
    uv run python scripts/build_manifest.py --version full-v1      # annotations only
    uv run python scripts/build_manifest.py --version full-v2 \\
        --frames-subset configs/data/kenai_subset.yaml            # + streamed Kenai frames
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
from passagewatch.ingestion.subsets import load_subset_config, select_clips
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
    parser.add_argument(
        "--frames-subset",
        type=Path,
        default=None,
        help="full manifests: a subset config whose streamed frames are validated too",
    )
    args = parser.parse_args(argv)

    out_dir = args.manifest_dir / "splits/cfc"
    try:
        manifest_paths(out_dir, args.version)
    except ValueError as exc:
        parser.error(str(exc))
    subset = args.version.split("-")[0]
    inventories = list(INVENTORIES[subset])
    report_name = subset
    inputs: dict[str, str] = {}
    if subset == "tiny":
        layout = CfcLayout.tiny(args.extract_dir)
    elif args.frames_subset is None:
        layout = CfcLayout.full(args.extract_dir)
    else:
        config = load_subset_config(args.frames_subset)
        full = CfcLayout.full(args.extract_dir)
        metadata = {i.location: full.metadata(i.location).clips for i in config.include}
        selected = frozenset(select_clips(config, metadata))
        layout = CfcLayout.full(args.extract_dir, args.extract_dir / config.name, selected)
        inventories.append(config.name)
        report_name = f"full-{config.name}"
        inputs["frames_subset_config"] = sha256_of(args.frames_subset)

    report = validate_dataset(layout)
    report_path = args.manifest_dir / f"validation/cfc/{report_name}.json"
    report.write_json(report_path)

    inputs["validation_report"] = sha256_of(report_path)
    for name in inventories:
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
        part = [r for r in rows if r["partition"] == partition]
        if part:
            with_frames = sum(r["frames_validated"] for r in part)
            print(
                f"  {partition:5} {len(part):5d} clips ({with_frames} with validated frames)  "
                f"tuning_allowed={part[0]['tuning_allowed']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
