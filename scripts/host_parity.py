"""Count the demo recordings with a release bundle on this host, with stage fingerprints.

Runs the worker's own path (``InferencePipeline`` from the bundle) on each frames ZIP in the
demo archive (docs/demos.md), and records, besides the counts, a SHA-256 of every stage's
output: the decoded frames, the network input (after preprocessing), the detections, and the
trajectories. Comparing two hosts' outputs shows whether they count alike and, if not, the
first stage where they differ. The detections are also saved (``.npz``) so the size of a
difference can be measured.

It uses only interfaces that release ``passagewatch-0.3.0`` already had, so it also runs
inside that release's image (``.github/workflows/host-parity.yml``).

Example:
    uv run python scripts/host_parity.py --archive dist/passagewatch-demos-v1.tar.gz \\
        --bundle bundles/passagewatch-0.3.0 --out runs/host-parity/mac.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from passagewatch.detection.classical import FrameDetections
from passagewatch.preprocessing.temporal import encode_frames
from passagewatch.service.catalog import ClipRecord
from passagewatch.service.media import iter_frames, probe
from passagewatch.service.worker.pipeline import InferencePipeline

COUNTING = {"policy": "cfc-compatible-v1", "line_x_normalized": 0.5}


def digest(arrays: list[Any]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def detections_digest(detections: list[FrameDetections], decimals: int | None) -> str:
    def r(a: Any) -> Any:
        return a if decimals is None else np.round(a, decimals)

    return digest([x for d in detections for x in (r(d.boxes), r(d.scores))])


def run_demo(pipeline: InferencePipeline, demo: dict[str, Any], media: Path) -> dict[str, Any]:
    info = probe(media, max_frames=100_000)
    meters = demo["meters"]
    clip = ClipRecord(
        clip_id=demo["demo_id"],
        source="host-parity",
        sha256=demo["sha256"],
        media_kind="frames",
        media_path=str(media),
        num_frames=info.num_frames,
        width=info.width,
        height=info.height,
        framerate=demo["framerate"],
        x_meter_start=meters["x_start"],
        x_meter_stop=meters["x_stop"],
        y_meter_start=meters["y_start"],
        y_meter_stop=meters["y_stop"],
        created_at="",
        expires_at=None,
        deleted_at=None,
    )
    version = pipeline.bundle.preprocessing_version
    decoded = list(iter_frames(media, "frames"))
    encoded = list(encode_frames(version, lambda: iter_frames(media, "frames")))
    detections = pipeline._detect(clip, iter(encoded), None)
    result = pipeline.run(clip, media, COUNTING)
    tracks = [x for t in result.trajectories for x in (t.frames, t.boxes, t.scores)]
    return {
        "counts": {"right": result.right, "left": result.left},
        "tracks": len(result.trajectories),
        "detections": int(sum(len(d.scores) for d in detections)),
        "sha256": {
            "decoded_frames": digest(decoded),
            "network_input": digest(encoded),
            "detections": detections_digest(detections, None),
            "detections_rounded_1e-2": detections_digest(detections, 2),
            "trajectories": digest(tracks),
        },
        "per_frame_decoded_sha256": [digest([f])[:16] for f in decoded],
        "per_frame_input_sha256": [digest([f])[:16] for f in encoded],
        "_detections": detections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    pipeline = InferencePipeline.load(args.bundle, device="cpu", batch_size=8, threads=args.threads)
    import cv2
    import onnxruntime
    import torch

    report: dict[str, Any] = {
        "bundle": pipeline.version,
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": onnxruntime.__version__,
            "torch": torch.__version__,
        },
        "demos": {},
    }
    arrays: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(args.archive, "r:gz") as tar:
        members = {m.name.split("/", 1)[1]: m for m in tar.getmembers() if "/" in m.name}
        catalog_file = tar.extractfile(members["catalog.json"])
        assert catalog_file is not None
        catalog = json.loads(catalog_file.read())
        for demo in catalog["demos"]:
            source = tar.extractfile(members[f"{demo['demo_id']}.zip"])
            assert source is not None
            data = source.read()
            if hashlib.sha256(data).hexdigest() != demo["sha256"]:
                raise SystemExit(f"{demo['demo_id']}: the ZIP does not match the catalog")
            media = Path(tmp) / f"{demo['demo_id']}.zip"
            media.write_bytes(data)
            result = run_demo(pipeline, demo, media)
            for i, d in enumerate(result.pop("_detections")):
                arrays[f"{demo['demo_id']}/{i}/boxes"] = d.boxes
                arrays[f"{demo['demo_id']}/{i}/scores"] = d.scores
            report["demos"][demo["demo_id"]] = result
            print(demo["demo_id"], result["counts"], result["tracks"], result["sha256"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
