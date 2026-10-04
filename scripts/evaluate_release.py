"""Evaluate a declared release (and its baselines) on one location of the test partition.

This is the frozen, one-time evaluation of the project plan. The release is fixed by its
committed manifest (``releases/manifests/<release>.json``); the bundle given must match it
exactly, or nothing runs. Nothing is selected: the bundle's checkpoint, threshold and tracker
are used as they are. ``--partition holdout`` runs the same code on kenai-holdout-v1, as a
rehearsal against numbers already known.

For every clip of ``--location`` whose frames are present under ``--frames-dir`` (one chunk
of recording days at a time), it records reference and predicted counts per system:

- ``release``: the release's detector (PyTorch on ``--device``), threshold and tracker;
- ``release_onnx``: for the first ``--parity-clips`` clips, the release's ONNX export on the
  CPU, the deployed runtime, to confirm the counts match;
- ``baseline``: optionally another release (``--baseline-bundle``, with its manifest);
- ``classical``: optionally the classical pipeline (``--classical-config``).

Results are appended to ``--out/<location>.json`` and clips already there are skipped, so a
location can be evaluated chunk by chunk and resumed. Clips with incomplete frames are
recorded as quarantined.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from passagewatch.counting.policy import CFC_COMPATIBLE_V1
from passagewatch.detection.onnx import onnx_detector
from passagewatch.evaluation.nmae import count_clip
from passagewatch.evaluation.release_eval import frames_problem, release_rows
from passagewatch.inference.classical import load_classical_config, run_clip
from passagewatch.inference.neural import (
    LoadedDetector,
    above,
    detect_clip,
    load_detector,
    track_and_evaluate,
)
from passagewatch.ingestion.cfc import CfcLayout, Clip, load_clip
from passagewatch.ingestion.metadata import load_metadata
from passagewatch.service.bundle import ReleaseBundle, load_bundle
from passagewatch.service.release import bundle_identity, read_manifest
from passagewatch.tracking.kalman import TrackerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def checked_bundle(bundle_dir: Path, manifest_path: Path) -> ReleaseBundle:
    """The bundle, refused unless it is exactly the release its manifest describes."""
    bundle = load_bundle(bundle_dir)
    manifest = read_manifest(manifest_path)
    if bundle.pipeline_version != manifest.release or bundle_identity(bundle) != manifest.bundle:
        raise SystemExit(f"{bundle_dir} is not the release described by {manifest_path}")
    return bundle


def neural_counts(detector: LoadedDetector, bundle: ReleaseBundle, clip: Clip) -> dict[str, Any]:
    detections = above(detect_clip(detector, clip), bundle.detector.score_threshold)
    tracker = TrackerConfig.model_validate(bundle.tracker.config)
    ev = track_and_evaluate(clip, detections, tracker)
    return {
        "predicted": [ev.error.predicted.right, ev.error.predicted.left],
        "tracks": ev.tracks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, default=None)
    parser.add_argument("--baseline-bundle", type=Path, default=None)
    parser.add_argument("--classical-config", type=Path, default=None)
    parser.add_argument("--manifest", default="full-v3")
    parser.add_argument("--partition", choices=["test", "holdout"], default="test")
    parser.add_argument("--location", required=True)
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parity-clips", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="at most N clips (rehearsal)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.baseline_manifest is None) != (args.baseline_bundle is None):
        parser.error("--baseline-bundle and --baseline-manifest go together")

    release = checked_bundle(args.bundle.resolve(), args.release_manifest)
    baseline = (
        checked_bundle(args.baseline_bundle.resolve(), args.baseline_manifest)
        if args.baseline_bundle is not None
        else None
    )
    classical = load_classical_config(args.classical_config) if args.classical_config else None
    device = torch.device(args.device)
    release_detector = load_detector(args.bundle / release.detector.checkpoint, device)
    onnx = None
    if args.parity_clips and release.detector.runtime == "onnxruntime":
        assert release.detector.onnx_file is not None
        onnx = onnx_detector(
            load_detector(args.bundle / release.detector.checkpoint, torch.device("cpu")),
            args.bundle / release.detector.onnx_file,
        )
    baseline_detector = (
        load_detector(args.baseline_bundle / baseline.detector.checkpoint, device)
        if baseline is not None and args.baseline_bundle is not None
        else None
    )

    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = release_rows(manifest_path, args.partition, args.location)
    metadata = load_metadata(
        args.extract_dir / "fish_counting_metadata/metadata" / f"{args.location}.json"
    ).clips
    out_path = args.out / f"{args.location}.json"
    report: dict[str, Any] = (
        json.loads(out_path.read_text(encoding="utf-8"))
        if out_path.exists()
        else {
            "release": release.pipeline_version,
            "release_manifest": str(args.release_manifest),
            "baseline": None if baseline is None else baseline.pipeline_version,
            "classical": None if classical is None else classical.name,
            "partition": args.partition,
            "location": args.location,
            "manifest": args.manifest,
            "device": str(device),
            "clips": {},
            "quarantined": {},
        }
    )
    if report["release"] != release.pipeline_version:
        raise SystemExit(f"{out_path} belongs to {report['release']}")
    present = [
        r
        for r in rows
        if (args.frames_dir / args.location / r["clip_name"]).is_dir()
        and r["clip_name"] not in report["clips"]
    ]
    present.sort(key=lambda r: r["clip_name"])
    if args.limit:
        present = present[: args.limit]
    parity_done = sum(1 for c in report["clips"].values() if "release_onnx" in c)
    layout = CfcLayout.full(
        args.extract_dir, args.frames_dir, frozenset(r["clip_name"] for r in present)
    )
    started = time.perf_counter()
    for i, row in enumerate(present, 1):
        name = row["clip_name"]
        meta = metadata[name]
        problem = frames_problem(args.frames_dir / args.location / name, meta.num_frames)
        if problem is not None:
            report["quarantined"][name] = problem
            continue
        clip = load_clip(layout, args.location, meta)
        reference = count_clip(clip.annotations, meta, CFC_COMPATIBLE_V1)
        entry: dict[str, Any] = {
            "reference": [reference.right, reference.left],
            "release": neural_counts(release_detector, release, clip),
        }
        if onnx is not None and parity_done < args.parity_clips:
            entry["release_onnx"] = neural_counts(onnx, release, clip)
            parity_done += 1
        if baseline is not None and baseline_detector is not None:
            entry["baseline"] = neural_counts(baseline_detector, baseline, clip)
        if classical is not None:
            run = run_clip(clip, classical)
            predicted = count_clip(run.tracks, meta, CFC_COMPATIBLE_V1)
            entry["classical"] = {
                "predicted": [predicted.right, predicted.left],
                "tracks": len(run.trajectories),
            }
        report["clips"][name] = entry
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        tmp.replace(out_path)
        print(
            f"\r  {args.location}: {i}/{len(present)} clips "
            f"({time.perf_counter() - started:.0f} s)",
            end="",
            file=sys.stderr,
        )
    print(file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(
        f"{args.location}: {len(report['clips'])} clips evaluated, "
        f"{len(report['quarantined'])} quarantined -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
