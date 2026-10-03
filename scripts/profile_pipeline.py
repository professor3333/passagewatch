"""Profile the worker's inference pipeline on one recording: time per stage and peak memory.

Runs exactly the service path (``InferencePipeline`` from a release bundle, reading frames
from an uploaded-style ZIP) on a CFC clip from a manifest, after one warm-up pass, and reports
seconds and milliseconds per frame for decode, preprocess, detect, postprocess, track, count
and review, plus the process's peak resident memory. ``--frames N`` builds a longer
recording by cycling the clip's frames, to check memory on long uploads (the service's
default upload limit is 6000 frames). The test partition cannot be selected.

Example:
    uv run python scripts/profile_pipeline.py --bundle bundles/passagewatch-0.2.0 \\
        --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1 \\
        --clip 2018-06-03-JD154_LeftNear_Stratum1_Set1_LN_2018-06-03_210000_2467_3008
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import torch

from passagewatch.inference.batch import make_layout, select_rows
from passagewatch.service.catalog import ClipRecord
from passagewatch.service.worker.pipeline import InferencePipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("decode", "preprocess", "detect", "postprocess", "track", "count", "review")


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Bytes on macOS, kilobytes on Linux.
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=REPO_ROOT / "bundles/active")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--partition", choices=["train", "val", "holdout"], default="val")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=None, help="cycle the clip to N frames")
    parser.add_argument("--no-warmup", action="store_true", help="skip the warm-up pass")
    parser.add_argument("--out", type=Path, default=None, help="write the profile as JSON")
    args = parser.parse_args(argv)

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = [r for r in select_rows(manifest_path, [args.partition]) if r["clip_name"] == args.clip]
    if not rows:
        parser.error(f"clip {args.clip} is not in {args.manifest}/{args.partition}")
    row = rows[0]
    layout = make_layout(args.manifest.split("-")[0], args.extract_dir, args.frames_dir, rows)
    meta = layout.metadata(row["location"]).clips[args.clip]
    frame_dir = layout.frame_dir(row["location"], args.clip)
    assert frame_dir is not None

    load_start = time.perf_counter()
    pipeline = InferencePipeline.load(args.bundle, device=args.device, batch_size=args.batch_size)
    load_s = time.perf_counter() - load_start
    with tempfile.TemporaryDirectory() as tmp:
        media = Path(tmp) / "clip.zip"
        num_frames = args.frames or meta.num_frames
        with zipfile.ZipFile(media, "w", zipfile.ZIP_STORED) as zf:
            for i in range(num_frames):
                zf.write(frame_dir / f"{i % meta.num_frames}.jpg", f"{i}.jpg")
        clip = ClipRecord(
            clip_id="profile",
            source="profile",
            sha256="0" * 64,
            media_kind="frames",
            media_path=str(media),
            num_frames=num_frames,
            width=meta.width,
            height=meta.height,
            framerate=meta.framerate,
            x_meter_start=meta.x_meter_start,
            x_meter_stop=meta.x_meter_stop,
            y_meter_start=meta.y_meter_start,
            y_meter_stop=meta.y_meter_stop,
            created_at="",
            expires_at=None,
            deleted_at=None,
        )
        counting = {"policy": "cfc-compatible-v1", "line_x_normalized": 0.5}
        if not args.no_warmup:
            pipeline.run(clip, media, counting)
        start = time.perf_counter()
        result = pipeline.run(clip, media, counting)
        total_s = time.perf_counter() - start

    frames = num_frames
    report = {
        "bundle": pipeline.version,
        "preprocessing_version": pipeline.bundle.preprocessing_version,
        "input": [pipeline.bundle.detector.input_height, pipeline.bundle.detector.input_width],
        "clip": args.clip,
        "frames": frames,
        "frame_size": [meta.width, meta.height],
        "device": args.device,
        "torch_threads": torch.get_num_threads(),
        "batch_size": args.batch_size,
        "host": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, "
        f"torch {torch.__version__}",
        "model_load_s": round(load_s, 3),
        "total_s": round(total_s, 3),
        "ms_per_frame": round(1000 * total_s / frames, 1),
        "frames_per_s": round(frames / total_s, 2),
        "stage_seconds": result.stage_seconds,
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    print(
        f"{report['bundle']} on {args.clip}: {frames} frames, {args.device}, "
        f"{report['torch_threads']} threads, batch {args.batch_size}"
    )
    print(
        f"  total {total_s:.2f} s = {report['ms_per_frame']} ms/frame "
        f"({report['frames_per_s']} frames/s); model load {load_s:.2f} s"
    )
    for stage in STAGES:
        seconds = result.stage_seconds.get(stage)
        if seconds is not None:
            print(
                f"  {stage:12} {seconds:7.2f} s  {1000 * seconds / frames:7.1f} ms/frame  "
                f"{100 * seconds / total_s:5.1f}%"
            )
    print(f"  peak resident memory {report['peak_rss_mb']} MB")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
