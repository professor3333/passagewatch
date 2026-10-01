"""Evaluate MOT-format tracking results by directional counting error (nMAE) per location.

Results must be laid out as CFC's: ``<results-dir>/<location>/<tracker>/data/<clip>.txt``.
Only kenai-val is evaluated by default. Test locations need ``--allow-test-locations``,
which is reserved for declared releases and for checking this evaluator against CFC's
published baselines; never use test results for tuning.

Example (reproduce the official numbers for CFC's published Baseline++ tracks):
    uv run python scripts/evaluate_counts.py --tracker baseline++ --allow-test-locations \\
        --locations kenai-val kenai-rightbank kenai-channel elwha nushagak
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from passagewatch.counting.policy import CFC_COMPATIBLE_V1
from passagewatch.evaluation.nmae import evaluate_mot_location, macro_nmae, summarize
from passagewatch.ingestion.cfc import LOCATIONS, CfcLayout
from passagewatch.ingestion.splits import Split, official_split

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO_ROOT / "data/extracted/cfc"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracker", required=True)
    parser.add_argument(
        "--results-dir", type=Path, default=EXTRACTED / "fish_counting_results/results"
    )
    parser.add_argument("--extract-dir", type=Path, default=EXTRACTED)
    parser.add_argument("--locations", nargs="+", choices=LOCATIONS, default=["kenai-val"])
    parser.add_argument("--allow-test-locations", action="store_true")
    parser.add_argument("--out", type=Path, default=None, help="write a JSON report here")
    args = parser.parse_args(argv)

    test = [loc for loc in args.locations if official_split(loc) is Split.TEST]
    if test and not args.allow_test_locations:
        parser.error(f"test locations {test} need --allow-test-locations")

    layout = CfcLayout.full(args.extract_dir)
    policy = CFC_COMPATIBLE_V1
    groups = []
    per_clip: dict[str, list[dict[str, object]]] = {}
    for location in args.locations:
        metadata = layout.metadata(location)
        if metadata.errors:
            print(f"error: invalid metadata for {location}: {metadata.errors}", file=sys.stderr)
            return 1
        errors = evaluate_mot_location(
            layout.annotations_dir / location,
            args.results_dir / location / args.tracker / "data",
            metadata.clips,
            policy,
        )
        groups.append(summarize(location, errors))
        per_clip[location] = [
            {
                "clip_name": e.clip_name,
                "reference": [e.reference.right, e.reference.left],
                "predicted": [e.predicted.right, e.predicted.left],
                "absolute_error": e.absolute_error,
            }
            for e in errors
        ]

    print(f"policy {policy.version}, tracker {args.tracker}")
    print(f"  {'location':16} {'clips':>5} {'error':>6} {'passages':>8} {'nMAE':>8}")
    for g in groups:
        value = "undef" if g.nmae is None else f"{g.nmae:.4f}"
        print(
            f"  {g.group:16} {g.clips:5d} {g.absolute_error:6d} {g.reference_passages:8d} "
            f"{value:>8}"
        )
    macro = macro_nmae(groups)
    macro_text = "undef" if macro is None else f"{macro:.4f}"
    print(f"  macro-average nMAE over these locations: {macro_text}")

    if args.out:
        report = {
            "policy": policy.version,
            "tracker": args.tracker,
            "locations": [
                {
                    "location": g.group,
                    "clips": g.clips,
                    "absolute_error": g.absolute_error,
                    "reference_passages": g.reference_passages,
                    "predicted_passages": g.predicted_passages,
                    "hours": g.hours,
                    "nmae": g.nmae,
                    "clip_errors": per_clip[g.group],
                }
                for g in groups
            ],
            "macro_nmae": macro,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
