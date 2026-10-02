"""Train a YOLOX detector on the train partition of a manifest; resumable.

Example (on a GPU host; see docs/training.md):
    python scripts/train_yolox.py --config configs/training/yolox-tiny-v1.yaml \\
        --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1 \\
        --out runs/train/yolox-tiny-v1 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from passagewatch.detection.neural import fetch_pretrained, load_pretrained_registry
from passagewatch.inference.batch import make_layout, select_rows
from passagewatch.training.data import FrameDataset
from passagewatch.training.yolox_train import Trainer, load_train_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--resume", action="store_true", help="continue from <out>/latest.pt")
    parser.add_argument("--max-iters", type=int, default=None, help="stop early (smoke runs)")
    parser.add_argument(
        "--background-cache",
        type=Path,
        default=REPO_ROOT / "data/cache/backgrounds",
        help="where temporal preprocessing caches each clip's background",
    )
    parser.add_argument("--pretrained-dir", type=Path, default=REPO_ROOT / "models/pretrained")
    args = parser.parse_args(argv)

    config = load_train_config(args.config)
    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = select_rows(manifest_path, ["train"])
    layout = make_layout(args.manifest.split("-")[0], args.extract_dir, args.frames_dir, rows)
    metadata_by_location = {
        loc: layout.metadata(loc).clips for loc in {r["location"] for r in rows}
    }
    clips = [(r["location"], metadata_by_location[r["location"]][r["clip_name"]]) for r in rows]

    dataset = FrameDataset(
        layout,
        clips,
        config.input_size,
        frame_stride=config.frame_stride,
        augment=config.augment,
        max_labels=config.max_labels,
        seed=config.seed,
        preprocessing=config.preprocessing,
        background_dir=args.background_cache,
    )
    weights = None
    if config.pretrained is not None:
        registry = load_pretrained_registry(REPO_ROOT / "configs/training/pretrained.yaml")
        weights = fetch_pretrained(registry[config.pretrained], args.pretrained_dir)

    sidecar = json.loads(manifest_path.with_suffix(".json").read_text(encoding="utf-8"))
    out = args.out or REPO_ROOT / "runs/train" / config.name
    trainer = Trainer(
        config,
        dataset,
        out,
        device=args.device,
        init_weights=weights,
        metadata={
            "manifest": args.manifest,
            "manifest_content_sha256": sidecar["content_sha256"],
            "partition": "train",
            "clips": len(clips),
            "pretrained": config.pretrained,
        },
    )
    latest = out / "latest.pt"
    if args.resume and latest.exists():
        trainer.resume(latest)
        print(f"resumed at epoch {trainer.state.epoch + 1}, iteration {trainer.state.iteration}")
    elif latest.exists():
        parser.error(f"{latest} exists; pass --resume to continue or choose another --out")
    print(
        f"{config.name}: {len(dataset)} samples from {len(clips)} clips, "
        f"{trainer.iters_per_epoch} iterations/epoch on {trainer.device}"
    )
    state = trainer.train(max_iters=args.max_iters)
    print(f"stopped at epoch {state.epoch + 1}, iteration {state.iteration} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
