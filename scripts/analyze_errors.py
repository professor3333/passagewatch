"""Explain counting errors on a development partition: passage outcomes, direction confusion,
detection recall by fish size, and nMAE by camera and stratum.

The system is either a neural detector's cached detections (``--detections`` with
``--threshold``, tracked with the tracker of ``--tracking-config``) or saved MOT tracks
(``--mot-dir``, e.g. from run_classical.py). The test partition cannot be selected.

Examples:
    uv run python scripts/analyze_errors.py --name yolox-tiny-v1-e25 \\
        --detections runs/neural/yolox-tiny-v1/epoch-025-7c9bfedc/val --threshold 0.2 \\
        --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1
    uv run python scripts/analyze_errors.py --name classical-v2 \\
        --mot-dir runs/classical/classical-v2-c4717f5f/full-v2/kenai-val/classical/data \\
        --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from passagewatch.counting.policy import CFC_COMPATIBLE_V1
from passagewatch.detection.classical import Scale, boxes_in_meters
from passagewatch.evaluation.errors import analyze_clip, combine, recall_by_size
from passagewatch.evaluation.nmae import ClipCountError, count_clip, summarize
from passagewatch.inference.batch import make_layout, select_rows
from passagewatch.inference.classical import load_classical_config
from passagewatch.inference.neural import above, load_detections
from passagewatch.ingestion.cfc import load_clip
from passagewatch.ingestion.mot import BoxAnnotations, read_mot
from passagewatch.tracking.kalman import KalmanTracker, trajectories_to_annotations

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZE_BINS_M2 = (0.02, 0.05, 0.1)
SIZE_LABELS = ("< 0.02 m²", "0.02-0.05 m²", "0.05-0.1 m²", ">= 0.1 m²")


def slice_of(clip_name: str) -> str:
    camera = re.search(r"_(Left|Right)(Far|Near)_Stratum(\d)", clip_name)
    return f"{camera[1]}{camera[2]} stratum {camera[3]}" if camera else "other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--partition", choices=["train", "val"], default="val")
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, default=None)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--detections", type=Path, help="evaluate_neural.py cache directory")
    source.add_argument("--mot-dir", type=Path, help="directory of <clip>.txt MOT tracks")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--tracking-config", type=Path, default=REPO_ROOT / "configs/tracking/classical-v2.yaml"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = select_rows(manifest_path, [args.partition])
    layout = make_layout(args.manifest.split("-")[0], args.extract_dir, args.frames_dir, rows)
    metadata = {loc: layout.metadata(loc).clips for loc in {r["location"] for r in rows}}
    tracker = load_classical_config(args.tracking_config).tracker
    policy = CFC_COMPATIBLE_V1

    clip_results = []
    errors: dict[str, list[ClipCountError]] = {}
    size_total: Counter[int] = Counter()
    size_found: Counter[int] = Counter()
    for row in rows:
        meta = metadata[row["location"]][row["clip_name"]]
        clip = load_clip(layout, row["location"], meta)
        scale = Scale(
            sx=1.0,
            sy=1.0,
            meters_per_px_x=abs(meta.x_meter_stop - meta.x_meter_start) / meta.width,
            meters_per_px_y=abs(meta.y_meter_stop - meta.y_meter_start) / meta.height,
        )
        if args.detections is not None:
            _, cached = load_detections(args.detections / row["location"] / f"{clip.name}.npz")
            detections = above(cached, args.threshold)
            centers = [boxes_in_meters(d.boxes, scale) for d in detections]
            predicted: BoxAnnotations = trajectories_to_annotations(
                KalmanTracker(tracker).run(detections, centers, frame_offset=clip.frame_start)
            )
            total, found = recall_by_size(
                clip.annotations,
                [d.boxes for d in detections],
                clip.frame_start,
                scale.meters_per_px_x * scale.meters_per_px_y,
                SIZE_BINS_M2,
            )
            size_total.update(total)
            size_found.update(found)
        else:
            predicted = read_mot(args.mot_dir / f"{clip.name}.txt")
        clip_results.append(
            analyze_clip(clip.name, clip.annotations, predicted, meta.width, meta.height, policy)
        )
        errors.setdefault(slice_of(clip.name), []).append(
            ClipCountError(
                clip.name,
                count_clip(clip.annotations, meta, policy),
                count_clip(predicted, meta, policy),
                clip.num_window_frames / meta.framerate,
            )
        )

    summary = combine(clip_results)
    slices = []
    for name, errs in sorted(errors.items()):
        g = summarize(name, errs)
        slices.append(
            {
                "slice": name,
                "clips": g.clips,
                "passages": g.reference_passages,
                "errors": g.absolute_error,
                "nmae": g.nmae,
            }
        )
    report: dict[str, Any] = {
        "system": args.name,
        "manifest": args.manifest,
        "partition": args.partition,
        "clips": len(rows),
        **summary,
        "slices": slices,
        "examples": [e for r in clip_results for e in r.examples],
    }
    if size_total:
        report["detection_recall_by_size"] = [
            {
                "size": SIZE_LABELS[b],
                "reference_boxes": size_total[b],
                "recall": size_found[b] / size_total[b] if size_total[b] else None,
            }
            for b in range(len(SIZE_LABELS))
        ]

    out = args.out or REPO_ROOT / "runs/errors" / f"{args.name}-{args.partition}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"{args.name} on {args.manifest}/{args.partition} ({len(rows)} clips) -> {out}")
    print("  reference passages:", report["reference_passages"])
    print("  predicted passages:", report["predicted_passages"])
    for ref, row in report["direction_confusion"].items():
        print(f"  {ref:16}", row)
    for s in slices:
        nmae = "undef" if s["nmae"] is None else f"{s['nmae']:.3f}"
        print(
            f"  {s['slice']:22} clips {s['clips']:3d} passages {s['passages']:4d} "
            f"errors {s['errors']:3d} nMAE {nmae}"
        )
    for r in report.get("detection_recall_by_size", []):
        recall = "—" if r["recall"] is None else f"{r['recall']:.2f}"
        print(f"  recall {r['size']:14} boxes {r['reference_boxes']:6d}  {recall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
