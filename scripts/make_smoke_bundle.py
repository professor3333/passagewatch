"""Create a release bundle with RANDOM weights, for deployment smoke tests only.

The real weights are not in git, so CI cannot use them. A randomly initialized YOLOX-Tiny
exercises everything a deployment must get right (image, entrypoints, volumes, bundle loading
and checksums, job flow) without claiming anything about counting quality. Its provenance
says so. It is built like the current release: temporal preprocessing, the network served
with ONNX Runtime, and calibration version review-v0.

Example:
    uv run python scripts/make_smoke_bundle.py --bundles-dir /tmp/pw-bundles
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

from passagewatch.calibration.versions import REVIEW_V0
from passagewatch.detection.neural import build_yolox
from passagewatch.detection.onnx import export_onnx
from passagewatch.inference.classical import load_classical_config
from passagewatch.preprocessing.temporal import TEMPORAL3
from passagewatch.service.bundle import activate, build_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "smoke-0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundles-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "random.pt"
        torch.save(
            {
                "config": {
                    "name": "smoke-random",
                    "model_size": "tiny",
                    "input_height": 320,
                    "input_width": 160,
                },
                "preprocessing_version": TEMPORAL3,
                "model": build_yolox("tiny").state_dict(),
                "state": {"epoch": 1},
                "metadata": {"warning": "random weights; smoke tests only"},
            },
            checkpoint,
        )
        export_onnx(checkpoint, Path(tmp) / "random.onnx")
        tracker = load_classical_config(REPO_ROOT / "configs/tracking/classical-v2.yaml").tracker
        path = build_bundle(
            checkpoint=checkpoint,
            tracker_config=tracker.model_dump(mode="json"),
            score_threshold=0.5,
            version=VERSION,
            bundles_dir=args.bundles_dir,
            provenance={"purpose": "SMOKE TEST ONLY: random weights, not a trained model"},
            calibration_version=REVIEW_V0,
            onnx=Path(tmp) / "random.onnx",
        )
    activate(args.bundles_dir, VERSION)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
