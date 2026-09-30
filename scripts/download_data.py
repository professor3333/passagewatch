"""Download, verify, extract, and inventory a named bundle of publisher data.

Example:
    uv run python scripts/download_data.py --bundle tiny
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from passagewatch.ingestion.download import download
from passagewatch.ingestion.extract import extract, is_archive
from passagewatch.ingestion.inventory import build_inventory, write_inventory
from passagewatch.ingestion.sources import load_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Progress:
    """Prints whole-percent progress for one file to stderr."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.last_percent = -1

    def __call__(self, done: int, total: int) -> None:
        percent = done * 100 // total
        if percent != self.last_percent:
            self.last_percent = percent
            end = "\n" if done >= total else ""
            print(f"\r  {self.label}: {percent:3d}%", end=end, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download, verify, extract, and inventory a bundle of publisher data."
    )
    parser.add_argument("--bundle", required=True, help="bundle name from the sources config")
    parser.add_argument("--sources", type=Path, default=REPO_ROOT / "configs/data/cfc_sources.yaml")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/cfc")
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument(
        "--inventory-dir", type=Path, default=REPO_ROOT / "data/manifests/inventory/cfc"
    )
    parser.add_argument("--no-extract", action="store_true", help="download and verify only")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    registry = load_registry(args.sources)
    bundle = registry.bundles.get(args.bundle)
    if bundle is None:
        parser.error(f"unknown bundle {args.bundle!r}; available: {sorted(registry.bundles)}")
    if bundle.contains_test_locations:
        logging.warning(
            "bundle %r contains official test-location clips; they must not be used for "
            "training, tuning, or hard-example mining",
            args.bundle,
        )

    for source in registry.bundle_files(args.bundle):
        path = download(source, args.raw_dir, progress=_Progress(source.key))
        if args.no_extract or not is_archive(path):
            continue
        extracted = extract(path, args.extract_dir, source_md5=source.md5)
        inventory_path = args.inventory_dir / f"{extracted.name}.parquet"
        if not inventory_path.exists():
            entries = build_inventory(extracted)
            write_inventory(entries, inventory_path)
            logging.info("inventoried %d files from %s", len(entries), extracted.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
