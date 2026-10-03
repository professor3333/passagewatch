"""Package a trained detector, tracker and counting policy as an immutable release bundle.

Example (the Stage 4 selection: yolox-tiny-v1 epoch 25, score threshold 0.2, classical-v2
tracker):
    uv run python scripts/build_bundle.py --version passagewatch-0.2.0 \\
        --checkpoint models/runs/yolox-tiny-v1/epoch-025.pt --score-threshold 0.2 \\
        --tracking-config configs/tracking/classical-v2.yaml \\
        --selection docs/neural_baseline.md --activate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from passagewatch.calibration.versions import CALIBRATIONS
from passagewatch.inference.classical import load_classical_config
from passagewatch.service.bundle import activate, build_bundle, load_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--tracking-config", type=Path, required=True)
    parser.add_argument("--selection", default=None, help="where the selection is documented")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="an ONNX export of the checkpoint (scripts/export_onnx.py); serve with ONNX Runtime",
    )
    parser.add_argument(
        "--calibration-version",
        choices=sorted(CALIBRATIONS),
        default=None,
        help="review scores, triage and audits (docs/review.md); none by default",
    )
    parser.add_argument("--bundles-dir", type=Path, default=REPO_ROOT / "bundles")
    parser.add_argument("--activate", action="store_true", help="make it the active release")
    args = parser.parse_args(argv)

    tracker = load_classical_config(args.tracking_config).tracker
    provenance = {"tracking_config": str(args.tracking_config)}
    if args.selection:
        provenance["selection"] = args.selection
    path = build_bundle(
        checkpoint=args.checkpoint,
        tracker_config=tracker.model_dump(mode="json"),
        score_threshold=args.score_threshold,
        version=args.version,
        bundles_dir=args.bundles_dir,
        provenance=provenance,
        calibration_version=args.calibration_version,
        onnx=args.onnx,
    )
    bundle = load_bundle(path)
    print(f"{bundle.pipeline_version}: config {bundle.config_sha256()[:12]} -> {path}")
    if args.activate:
        print(f"active -> {activate(args.bundles_dir, args.version)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
