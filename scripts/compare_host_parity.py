"""Compare two ``scripts/host_parity.py`` outputs: counts, stage fingerprints, detections.

Detections are matched one-to-one within each frame (greedily, by the largest box-corner
distance, up to ``--max-px`` pixels). For matched pairs it reports the score and box
differences; unmatched detections are listed with their scores, which shows whether they
sit near the detector's score threshold.

Example:
    uv run python scripts/compare_host_parity.py runs/host-parity/mac-m1.json \\
        runs/host-parity/linux-amd64-image.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def match_frame(
    a_boxes: Any, a_scores: Any, b_boxes: Any, b_scores: Any, max_px: float
) -> tuple[list[float], list[float], list[float]]:
    """(score differences, box differences, unmatched scores) of one frame."""
    pairs = []
    for i in range(len(a_scores)):
        for j in range(len(b_scores)):
            dist = float(np.abs(a_boxes[i] - b_boxes[j]).max())
            if dist <= max_px:
                pairs.append((dist, i, j))
    used_a: set[int] = set()
    used_b: set[int] = set()
    scores, boxes = [], []
    for dist, i, j in sorted(pairs):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        scores.append(abs(float(a_scores[i]) - float(b_scores[j])))
        boxes.append(dist)
    unmatched = [float(a_scores[i]) for i in range(len(a_scores)) if i not in used_a]
    unmatched += [float(b_scores[j]) for j in range(len(b_scores)) if j not in used_b]
    return scores, boxes, unmatched


def compare(a_path: Path, b_path: Path, max_px: float) -> dict[str, Any]:
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (a_path, b_path))
    a_npz, b_npz = (np.load(p.with_suffix(".npz")) for p in (a_path, b_path))
    out: dict[str, Any] = {}
    for demo, ra in a["demos"].items():
        rb = b["demos"][demo]
        frames = len(ra["per_frame_decoded_sha256"])
        scores, boxes, unmatched = [], [], []
        for i in range(frames):
            s, bx, u = match_frame(
                a_npz[f"{demo}/{i}/boxes"],
                a_npz[f"{demo}/{i}/scores"],
                b_npz[f"{demo}/{i}/boxes"],
                b_npz[f"{demo}/{i}/scores"],
                max_px,
            )
            scores += s
            boxes += bx
            unmatched += u
        out[demo] = {
            "counts": [ra["counts"], rb["counts"]],
            "tracks": [ra["tracks"], rb["tracks"]],
            "same": {k: ra["sha256"][k] == rb["sha256"][k] for k in ra["sha256"]},
            "matched_detections": len(scores),
            "score_difference": {
                "median": float(np.median(scores)) if scores else 0.0,
                "max": float(max(scores, default=0.0)),
            },
            "box_difference_px_max": float(max(boxes, default=0.0)),
            "unmatched_scores": sorted(round(s, 3) for s in unmatched),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--max-px", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(compare(args.a, args.b, args.max_px), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
