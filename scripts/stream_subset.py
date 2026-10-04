"""Stream a large CFC imagery archive and keep only the clips of a named subset.

The archive is never stored. Its MD5 is still verified, and the kept frames are promoted
only if it matches. Afterwards the frames are inventoried with SHA-256, like every other
extraction.

Example:
    uv run python scripts/stream_subset.py --config configs/data/kenai_subset.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from passagewatch.ingestion.cfc import CfcLayout
from passagewatch.ingestion.inventory import build_inventory, write_inventory
from passagewatch.ingestion.sources import load_registry
from passagewatch.ingestion.stream_extract import stream_extract
from passagewatch.ingestion.subsets import (
    CfcFrameSelector,
    archive_locations,
    load_subset_config,
    select_clips,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Progress:
    """Logs progress every few percent, with throughput, for a long background run."""

    def __init__(self, step: int = 2) -> None:
        self.step = step
        self.last = -step
        self.start = time.monotonic()

    def __call__(self, done: int, total: int) -> None:
        percent = done * 100 // total
        if percent >= self.last + self.step or done >= total:
            self.last = percent
            elapsed = time.monotonic() - self.start
            rate = done / elapsed / 1e6 if elapsed > 0 else 0.0
            eta = (total - done) / (rate * 1e6) / 60 if rate > 0 else float("nan")
            print(
                f"{percent:3d}%  {done / 1e9:6.1f}/{total / 1e9:.1f} GB  {rate:5.1f} MB/s  "
                f"eta {eta:5.0f} min",
                file=sys.stderr,
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=REPO_ROOT / "configs/data/cfc_sources.yaml")
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument(
        "--inventory-dir", type=Path, default=REPO_ROOT / "data/manifests/inventory/cfc"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = load_subset_config(args.config)
    source = load_registry(args.sources).resolve(config.archive)
    layout = CfcLayout.full(args.extract_dir)
    metadata = {}
    for location in {i.location for i in config.include}:
        loaded = layout.metadata(location)
        if loaded.errors:
            parser.error(f"invalid metadata for {location}: {loaded.errors}")
        metadata[location] = loaded.clips
    selected = select_clips(config, metadata)
    all_clips = {
        name
        for location in archive_locations(config.archive)
        for name in layout.metadata(location).clips
    }
    expected = sum(metadata[loc][clip].num_frames for clip, loc in selected.items())
    print(
        f"{config.name}: keeping {len(selected)} clips ({expected} frames) from {source.ref} "
        f"({source.size / 1e9:.1f} GB streamed)",
        file=sys.stderr,
    )

    selector = CfcFrameSelector(config.member_prefix, selected, all_clips)
    dest = args.extract_dir / config.name
    result = stream_extract(
        source, dest, selector, description=config.describe(), progress=_Progress()
    )
    summary = {
        "files_written": result.files_written,
        "bytes_written": result.bytes_written,
        "skipped": dict(selector.skipped),
        "unknown_examples": selector.unknown_examples,
        "expected_frames": expected,
    }
    print(json.dumps(summary, indent=1), file=sys.stderr)

    inventory_path = args.inventory_dir / f"{config.name}.parquet"
    if not inventory_path.exists():
        entries = build_inventory(dest)
        write_inventory(entries, inventory_path)
        print(f"inventoried {len(entries)} files -> {inventory_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
