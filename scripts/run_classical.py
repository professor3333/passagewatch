"""Run the classical baseline over manifest clips: frames -> tracks -> counts -> report.

Writes MOT-format tracks (1-based, CFC layout ``<location>/<tracker>/data/<clip>.txt``) and
``report.json`` with per-location nMAE, per-clip counts, the config and its hash, and
runtimes. Only train and val partitions can be run; test locations are refused.

Examples:
    uv run python scripts/run_classical.py --manifest tiny-v1 --partitions val
    uv run python scripts/run_classical.py --manifest full-v2 --frames-dir \\
        data/extracted/cfc/kenai-dev-v1 --partitions train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from passagewatch.counting.policy import CFC_COMPATIBLE_V1
from passagewatch.evaluation.nmae import ClipCountError, count_clip, macro_nmae, summarize
from passagewatch.inference.classical import load_classical_config, run_clip
from passagewatch.ingestion.cfc import CfcLayout, load_clip
from passagewatch.ingestion.manifest import read_manifest
from passagewatch.ingestion.mot import write_mot

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER = "classical"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="manifest version, e.g. tiny-v1")
    parser.add_argument("--partitions", nargs="+", choices=["train", "val"], default=["val"])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/tracking/classical-v1.yaml"
    )
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, default=None, help="full manifests only")
    parser.add_argument("--limit", type=int, default=None, help="at most N clips (smoke runs)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = [
        r
        for r in read_manifest(manifest_path)
        if r["partition"] in args.partitions and r["usable"] and r["frames_validated"]
    ]
    assert all(r["tuning_allowed"] for r in rows)
    rows = rows[: args.limit] if args.limit else rows
    if not rows:
        parser.error("no usable clips with validated frames in the selected partitions")

    if args.manifest.startswith("tiny"):
        layout = CfcLayout.tiny(args.extract_dir)
    else:
        if args.frames_dir is None:
            parser.error("--frames-dir is required for full manifests")
        layout = CfcLayout.full(
            args.extract_dir, args.frames_dir, frozenset(r["clip_name"] for r in rows)
        )
    config = load_classical_config(args.config)
    digest = config.sha256()
    out = args.out or REPO_ROOT / "runs/classical" / f"{config.name}-{digest[:8]}" / args.manifest
    policy = CFC_COMPATIBLE_V1

    metadata = {loc: layout.metadata(loc).clips for loc in {r["location"] for r in rows}}
    errors: dict[str, list[ClipCountError]] = {}
    clips_out = []
    seconds = {"decode": 0.0, "detect": 0.0, "track": 0.0}
    frames = 0
    started = time.perf_counter()
    for i, row in enumerate(rows, 1):
        meta = metadata[row["location"]][row["clip_name"]]
        clip = load_clip(layout, row["location"], meta)
        result = run_clip(clip, config)
        write_mot(result.tracks, out / row["location"] / TRACKER / "data" / f"{clip.name}.txt")
        reference = count_clip(clip.annotations, meta, policy)
        predicted = count_clip(result.tracks, meta, policy)
        error = ClipCountError(
            clip.name, reference, predicted, clip.num_window_frames / meta.framerate
        )
        errors.setdefault(row["location"], []).append(error)
        for key, value in result.seconds.items():
            seconds[key] += value
        frames += clip.num_window_frames
        clips_out.append(
            {
                "location": row["location"],
                "clip_name": clip.name,
                "partition": row["partition"],
                "frames": clip.num_window_frames,
                "detections": result.detections,
                "tracks": len(result.trajectories),
                "reference": [reference.right, reference.left],
                "predicted": [predicted.right, predicted.left],
                "absolute_error": error.absolute_error,
            }
        )
        print(
            f"[{i}/{len(rows)}] {row['location']} {clip.name}: ref {reference.right}R "
            f"{reference.left}L, pred {predicted.right}R {predicted.left}L",
            file=sys.stderr,
        )

    groups = [summarize(loc, errs) for loc, errs in errors.items()]
    report = {
        "pipeline": TRACKER,
        "config": config.model_dump(mode="json"),
        "config_sha256": digest,
        "counting_policy": policy.version,
        "line_x_normalized": policy.line_x_normalized,
        "manifest": args.manifest,
        "partitions": args.partitions,
        "note": (
            "tiny subset: 50-frame windows with truncated trajectories; not comparable "
            "with benchmark numbers"
            if args.manifest.startswith("tiny")
            else ""
        ),
        "locations": [
            {
                "location": g.group,
                "clips": g.clips,
                "absolute_error": g.absolute_error,
                "reference_passages": g.reference_passages,
                "predicted_passages": g.predicted_passages,
                "nmae": g.nmae,
            }
            for g in groups
        ],
        "macro_nmae": macro_nmae(groups),
        "runtime": {
            "frames": frames,
            "seconds": {k: round(v, 2) for k, v in seconds.items()},
            "wall_seconds": round(time.perf_counter() - started, 2),
        },
        "clips": clips_out,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"{config.name} ({digest[:8]}) on {args.manifest} {args.partitions} -> {out}")
    for g in groups:
        nmae_text = "undef" if g.nmae is None else f"{g.nmae:.4f}"
        print(
            f"  {g.group:12} clips {g.clips:4d}  error {g.absolute_error:5d}  "
            f"passages {g.reference_passages:5d}  predicted {g.predicted_passages:5d}  "
            f"nMAE {nmae_text}"
        )
    per_frame = 1000 * sum(seconds.values()) / max(1, frames)
    print(f"  {frames} frames, {per_frame:.1f} ms/frame (decode+detect+track)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
