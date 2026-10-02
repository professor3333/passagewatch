"""Run the classical baseline over manifest clips: frames -> tracks -> counts -> report.

Writes MOT-format tracks (1-based, CFC layout ``<location>/classical/data/<clip>.txt``) and
``report.json`` with per-location nMAE, detection recall/precision, per-clip counts, the
config and its hash, and runtimes. Train, val or (on its own) the internal holdout can be
run; the test partition cannot.

Examples:
    uv run python scripts/run_classical.py --manifest tiny-v1 --partitions val
    uv run python scripts/run_classical.py --manifest full-v2 --partitions val \\
        --frames-dir data/extracted/cfc/kenai-dev-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from passagewatch.counting.policy import CFC_COMPATIBLE_V1
from passagewatch.inference.batch import (
    ClipOutcome,
    clip_records,
    default_workers,
    make_layout,
    make_tasks,
    run_tasks,
    select_rows,
    summarize_outcomes,
)
from passagewatch.inference.classical import load_classical_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(title)
    for g in summary["locations"]:
        nmae = "undef" if g["nmae"] is None else f"{g['nmae']:.4f}"
        recall = g["detection_recall"]
        precision = g["detection_precision"]
        print(
            f"  {g['location']:12} clips {g['clips']:4d}  error {g['absolute_error']:5d}  "
            f"passages {g['reference_passages']:5d}  predicted {g['predicted_passages']:5d}  "
            f"nMAE {nmae}  det recall {recall or 0:.2f} precision {precision or 0:.2f}"
        )
    runtime = summary["runtime"]
    print(f"  {runtime['frames']} frames, {runtime['ms_per_frame']} ms/frame CPU")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="manifest version, e.g. tiny-v1")
    parser.add_argument(
        "--partitions", nargs="+", choices=["train", "val", "holdout"], default=["val"]
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/tracking/classical-v1.yaml"
    )
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, default=None, help="full manifests only")
    parser.add_argument("--limit", type=int, default=None, help="at most N clips (smoke runs)")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = select_rows(manifest_path, args.partitions, args.limit)
    if not rows:
        parser.error("no usable clips with validated frames in the selected partitions")
    try:
        subset = args.manifest.split("-")[0]
        layout = make_layout(subset, args.extract_dir, args.frames_dir, rows)
    except ValueError as exc:
        parser.error(str(exc))
    config = load_classical_config(args.config)
    digest = config.sha256()
    out = args.out or REPO_ROOT / "runs/classical" / f"{config.name}-{digest[:8]}" / args.manifest

    def progress(i: int, outcome: ClipOutcome) -> None:
        e = outcome.error
        print(
            f"[{i}/{len(rows)}] {outcome.location} {e.clip_name}: ref {e.reference.right}R "
            f"{e.reference.left}L, pred {e.predicted.right}R {e.predicted.left}L",
            file=sys.stderr,
        )

    outcomes = run_tasks(make_tasks(rows, layout, config, out), args.workers, progress)
    summary = summarize_outcomes(outcomes)
    report = {
        "pipeline": "classical",
        "config": config.model_dump(mode="json"),
        "config_sha256": digest,
        "counting_policy": CFC_COMPATIBLE_V1.version,
        "line_x_normalized": CFC_COMPATIBLE_V1.line_x_normalized,
        "manifest": args.manifest,
        "partitions": args.partitions,
        "note": (
            "tiny subset: 50-frame windows with truncated trajectories; not comparable "
            "with benchmark numbers"
            if args.manifest.startswith("tiny")
            else ""
        ),
        **summary,
        "clips": clip_records(outcomes),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print_summary(
        f"{config.name} ({digest[:8]}) on {args.manifest} {args.partitions} -> {out}", summary
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
