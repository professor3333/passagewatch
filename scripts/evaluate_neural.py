"""Evaluate trained YOLOX checkpoints by directional counting nMAE, through the same tracker.

1. Detect: for every listed epoch and every clip, detections are computed once (score >=
   0.05) and cached under ``runs/neural/<run>/epoch-NNN-<sha8>/<partition>/``; existing
   cache files are reused, so an interrupted run resumes.
2. Count: for every (epoch, score threshold), detections above the threshold go through the
   Kalman tracker of the given tracking config and the cfc-compatible-v1 counting policy.

Writes ``report.json`` with every combination and prints a table. The test partition
cannot be selected.

Example:
    uv run python scripts/evaluate_neural.py --run models/runs/yolox-tiny-v1 \\
        --epochs 20 25 30 --thresholds 0.1 0.2 0.3 0.4 0.5 \\
        --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1 --partition val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from passagewatch.detection.neural import select_device
from passagewatch.detection.onnx import onnx_detector
from passagewatch.evaluation.detection import DetectionMatch
from passagewatch.evaluation.nmae import macro_nmae, summarize
from passagewatch.inference.batch import make_layout, select_rows
from passagewatch.inference.classical import load_classical_config
from passagewatch.inference.neural import (
    above,
    detect_clip,
    file_sha256,
    load_detections,
    load_detector,
    save_detections,
    track_and_evaluate,
)
from passagewatch.ingestion.cfc import load_clip

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="training run directory")
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--partition",
        choices=["train", "val", "holdout"],
        default="val",
        help="holdout: evaluates one already chosen epoch and threshold, never selects them",
    )
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument(
        "--tracking-config",
        type=Path,
        default=REPO_ROOT / "configs/tracking/classical-v2.yaml",
        help="its tracker section is used (same tracker as the classical baseline)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="run the network with ONNX Runtime from this export of the (single) epoch's "
        "checkpoint; its detections are cached separately",
    )
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (ONNX/torch)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.onnx is not None and len(args.epochs) != 1:
        parser.error("--onnx goes with exactly one --epochs value (the exported checkpoint)")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if args.partition == "holdout" and (len(args.epochs) != 1 or len(args.thresholds) != 1):
        parser.error("on the holdout, pass exactly one --epochs and one --thresholds value")

    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = select_rows(manifest_path, [args.partition])
    layout = make_layout(args.manifest.split("-")[0], args.extract_dir, args.frames_dir, rows)
    metadata = {loc: layout.metadata(loc).clips for loc in {r["location"] for r in rows}}
    clips = [
        load_clip(layout, r["location"], metadata[r["location"]][r["clip_name"]]) for r in rows
    ]
    tracker = load_classical_config(args.tracking_config).tracker
    device = select_device(args.device)
    out = args.out or REPO_ROOT / "runs/neural" / args.run.name
    frames = sum(c.num_window_frames for c in clips)

    results: list[dict[str, Any]] = []
    for epoch in args.epochs:
        detector = load_detector(args.run / f"epoch-{epoch:03d}.pt", device)
        name = f"epoch-{epoch:03d}-{detector.checkpoint_sha256[:8]}"
        if args.onnx is not None:
            detector = onnx_detector(detector, args.onnx, args.threads)
            name += f"-onnx-{file_sha256(args.onnx)[:8]}"
        elif device.type != "mps":
            # Devices can differ in the last digits of scores; MPS caches keep the plain name.
            name += f"-{device.type}"
        cache = out / name / args.partition
        started, computed = time.perf_counter(), 0
        for i, clip in enumerate(clips, 1):
            path = cache / clip.location / f"{clip.name}.npz"
            if not path.exists():
                save_detections(path, clip.frame_start, detect_clip(detector, clip))
                computed += clip.num_window_frames
            print(f"\r  epoch {epoch}: detections {i}/{len(clips)} clips", end="", file=sys.stderr)
        elapsed = time.perf_counter() - started
        print(file=sys.stderr)
        if computed:
            print(f"epoch {epoch}: {1000 * elapsed / computed:.1f} ms/frame on {device}")

        cached = {c.name: load_detections(cache / c.location / f"{c.name}.npz")[1] for c in clips}
        for threshold in args.thresholds:
            evaluations = [
                track_and_evaluate(c, above(cached[c.name], threshold), tracker) for c in clips
            ]
            by_location: dict[str, list[Any]] = {}
            for clip, ev in zip(clips, evaluations, strict=True):
                by_location.setdefault(clip.location, []).append(ev)
            groups = [summarize(loc, [e.error for e in evs]) for loc, evs in by_location.items()]
            match = sum((e.detection for e in evaluations), DetectionMatch(0, 0, 0))
            missed = sum(
                max(0, e.error.reference.right - e.error.predicted.right)
                + max(0, e.error.reference.left - e.error.predicted.left)
                for e in evaluations
            )
            results.append(
                {
                    "epoch": epoch,
                    "threshold": threshold,
                    "checkpoint_sha256": detector.checkpoint_sha256,
                    "runtime": "onnxruntime" if args.onnx is not None else "torch",
                    "macro_nmae": macro_nmae(groups),
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
                    "missed_passages": missed,
                    "false_passages": sum(g.absolute_error for g in groups) - missed,
                    "detection_recall": match.recall,
                    "detection_precision": match.precision,
                    "clips": [
                        {
                            "clip_name": e.error.clip_name,
                            "reference": [e.error.reference.right, e.error.reference.left],
                            "predicted": [e.error.predicted.right, e.error.predicted.left],
                            "tracks": e.tracks,
                        }
                        for e in evaluations
                    ],
                }
            )
            r = results[-1]
            print(
                f"  epoch {epoch:3d}  threshold {threshold:.2f}  nMAE {r['macro_nmae']:.4f}  "
                f"missed {missed:3d}  false {r['false_passages']:3d}  "
                f"det recall {match.recall or 0:.2f} precision {match.precision or 0:.2f}"
            )

    best = min(results, key=lambda r: (r["macro_nmae"], r["epoch"], -r["threshold"]))
    report = {
        "run": args.run.name,
        "manifest": args.manifest,
        "partition": args.partition,
        "frames": frames,
        "tracker": tracker.model_dump(mode="json"),
        "tracking_config": str(args.tracking_config.relative_to(REPO_ROOT)),
        "selection_rule": "lowest macro nMAE; ties: earlier epoch, then higher threshold",
        "best": {k: best[k] for k in ("epoch", "threshold", "macro_nmae", "checkpoint_sha256")},
        "results": results,
    }
    out.mkdir(parents=True, exist_ok=True)
    suffix = (
        "-onnx" if args.onnx is not None else ("" if device.type == "mps" else f"-{device.type}")
    )
    report_path = out / f"report-{args.partition}{suffix}.json"
    report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(
        f"best: epoch {best['epoch']}, threshold {best['threshold']}, "
        f"nMAE {best['macro_nmae']:.4f} -> {report_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
