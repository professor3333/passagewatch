"""Validate extracted CFC data and write a JSON report of issues and quarantined clips.

Examples:
    uv run python scripts/validate_data.py --subset tiny
    uv run python scripts/validate_data.py --subset full        # annotations + metadata only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from passagewatch.ingestion.cfc import CfcLayout
from passagewatch.validation.cfc import validate_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subset", choices=["tiny", "full"], required=True)
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="full subset only: <location>/<clip>/<n>.jpg frames to check as well",
    )
    parser.add_argument("--no-images", action="store_true", help="skip decoding frames")
    parser.add_argument("--image-size-tolerance-px", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: data/manifests/validation/cfc/<subset>.json",
    )
    args = parser.parse_args(argv)

    if args.subset == "tiny":
        layout = CfcLayout.tiny(args.extract_dir)
    else:
        layout = CfcLayout.full(args.extract_dir, frames_dir=args.frames_dir)
    report = validate_dataset(
        layout,
        check_images=not args.no_images,
        image_size_tolerance_px=args.image_size_tolerance_px,
    )
    out = args.out or REPO_ROOT / f"data/manifests/validation/cfc/{args.subset}.json"
    report.write_json(out)

    summary = report.summary()
    print(f"{summary['clips']} clips, {summary['quarantined']} quarantined -> {out}")
    for location, statuses in summary["by_location"].items():
        print(f"  {location:16} " + "  ".join(f"{k}={v}" for k, v in statuses.items()))
    for location, issues in report.location_issues.items():
        for issue in issues:
            print(f"  {location}: {issue.severity}: {issue.code}: {issue.message}")
    for code, totals in summary["issue_totals"].items():
        print(f"  {code:30} {totals['count']:6d} in {totals['clips']} clip(s)")
    for clip in report.quarantined:
        print(f"  QUARANTINED {clip.location}/{clip.clip_name}: {clip.quarantine_reasons}")
    return 0 if not report.location_issues else 1


if __name__ == "__main__":
    sys.exit(main())
