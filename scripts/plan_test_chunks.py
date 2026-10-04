"""Plan how to stream one test location in chunks of whole recording days that fit a disk.

The CFC archives interleave frames from all clips, so a clip is complete only once its whole
archive has been read. Each chunk is therefore a subset config (``configs/data``-style) that
keeps some recording days; the archive is streamed once per chunk. Bytes per frame are
estimated from the archive's size and the location's frame count. Writes one YAML file per
chunk to ``--out-dir`` and prints their paths.

Example:
    uv run python scripts/plan_test_chunks.py --location kenai-rightbank --budget-gb 25 \\
        --out-dir /tmp/chunks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from passagewatch.evaluation.release_eval import plan_day_chunks
from passagewatch.ingestion.metadata import load_metadata
from passagewatch.ingestion.sources import load_registry
from passagewatch.ingestion.splits import Split, official_split
from passagewatch.ingestion.subsets import ANY_TOP_DIRECTORY, ARCHIVE_LOCATIONS

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--location", required=True)
    parser.add_argument("--budget-gb", type=float, default=25.0)
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--sources", type=Path, default=REPO_ROOT / "configs/data/cfc_sources.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if official_split(args.location) is not Split.TEST:
        parser.error(f"{args.location} is not a test location")
    archive = next(k for k, locs in ARCHIVE_LOCATIONS.items() if args.location in locs)
    ref = f"g945x-41103/{archive}"
    size = load_registry(args.sources).resolve(ref).size
    clips = load_metadata(
        args.extract_dir / "fish_counting_metadata/metadata" / f"{args.location}.json"
    ).clips
    frames = sum(m.num_frames for m in clips.values())
    chunks = plan_day_chunks(clips, size / frames, args.budget_gb * 1e9)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, days in enumerate(chunks, 1):
        config = {
            "name": f"test-{args.location}-{i}",
            "archive": ref,
            "member_prefix": ANY_TOP_DIRECTORY,
            "include": [{"location": args.location, "days": [d.isoformat() for d in days]}],
        }
        path = args.out_dir / f"{config['name']}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(path)
    print(
        f"{args.location}: {len(clips)} clips, {frames} frames, ~{size / 1e9:.1f} GB in "
        f"{len(chunks)} chunk(s) of at most {args.budget_gb} GB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
