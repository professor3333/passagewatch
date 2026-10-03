"""Export a trained detector checkpoint to ONNX and check its outputs against PyTorch.

Writes ``--out`` (fixed input size, dynamic batch), then compares the PyTorch and ONNX
Runtime outputs on a batch of real frames and prints the largest differences. The counting
check is ``evaluate_neural.py --onnx`` on development data.

Example:
    uv run python scripts/export_onnx.py --checkpoint models/runs/yolox-tiny-v1/epoch-025.pt \\
        --out models/onnx/yolox-tiny-v1-epoch-025.onnx \\
        --frames data/extracted/cfc/kenai-dev-v1/kenai-val/<clip>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from passagewatch.detection.onnx import export_onnx, onnx_detector
from passagewatch.inference.neural import file_sha256
from passagewatch.preprocessing.letterbox import letterbox_image, to_network_input
from passagewatch.preprocessing.temporal import encode_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True, help="a clip's frame directory")
    parser.add_argument("--count", type=int, default=8, help="frames to compare")
    args = parser.parse_args(argv)

    detector = export_onnx(args.checkpoint, args.out)
    paths = sorted(args.frames.glob("*.jpg"), key=lambda p: int(p.stem))[: args.count + 1]
    frames = [np.asarray(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]
    images = list(encode_frames(detector.preprocessing, lambda: iter(frames)))[: args.count]
    batch = to_network_input(np.stack([letterbox_image(i, detector.input_size)[0] for i in images]))
    with torch.no_grad():
        expected = detector.model(batch).float()
    actual = onnx_detector(detector, args.out).model(batch)
    print(
        f"{args.out} ({args.out.stat().st_size / 1e6:.1f} MB, sha256 {file_sha256(args.out)[:12]})"
    )
    print(f"  box max |diff| {float((expected[..., :4] - actual[..., :4]).abs().max()):.2e} px")
    print(f"  score max |diff| {float((expected[..., 4:] - actual[..., 4:]).abs().max()):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
