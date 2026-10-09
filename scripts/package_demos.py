"""Build the demo examples' frames archive (docs/demos.md) from the local CFC data.

For each demo in ``demos/catalog.json``, reads the clip's frames, metadata and MOT
annotations, writes its frames as a reproducible ZIP, and checks that the catalog's
SHA-256, frame count, frame rate, sonar window and reference counts (the annotations counted
with ``cfc-compatible-v1``) match. With ``--update`` the catalog is rewritten from the data
instead. Then ``dist/passagewatch-<version>.tar.gz`` is written with the ZIPs, the catalog
and the license notices, for attaching to a GitHub release.

Frames are looked up under ``<frames-root>/<subset>/<location>/<clip>/``, for example
``data/extracted/cfc/kenai-holdout-v1/kenai-train/...`` (``make data-kenai-holdout``) and
``data/extracted/cfc/demo-channel-v1/kenai-channel/...``
(``scripts/stream_subset.py --config configs/data/demo_channel.yaml``).

Example:
    uv run python scripts/package_demos.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from passagewatch.counting.policy import CFC_COMPATIBLE_V1
from passagewatch.evaluation.nmae import count_clip
from passagewatch.ingestion.cfc import CfcLayout, frame_path, load_clip
from passagewatch.service.demos import (
    DemoCatalog,
    DemoEntry,
    catalog_json,
    frames_zip,
    load_catalog,
    pack_demos,
    sha256_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = REPO_ROOT / "data/extracted/cfc"


def find_frames(root: Path, location: str, clip: str) -> Path:
    found = sorted(p for p in root.glob(f"*/{location}/{clip}") if p.is_dir())
    if len(found) != 1:
        raise SystemExit(f"expected one frame directory for {location}/{clip}, found {found}")
    return found[0].parent.parent


def from_data(demo: DemoEntry, root: Path, zips: Path) -> DemoEntry:
    """The entry as the data says it should be; its ZIP is written to ``zips``."""
    location, name = demo.source.location, demo.source.clip_name
    layout = CfcLayout.full(EXTRACTED, frames_dir=find_frames(root, location, name))
    meta = layout.metadata(location).clips[name]
    clip = load_clip(layout, location, meta)
    assert clip.frame_dir is not None
    data = frames_zip([frame_path(clip.frame_dir, i) for i in range(meta.num_frames)])
    (zips / demo.zip_name).write_bytes(data)
    counts = count_clip(clip.annotations, meta, CFC_COMPATIBLE_V1)
    return DemoEntry.model_validate(
        demo.model_dump(mode="json")
        | {
            "reference": {"right": counts.right, "left": counts.left},
            "framerate": meta.framerate,
            "meters": {
                "x_start": meta.x_meter_start,
                "x_stop": meta.x_meter_stop,
                "y_start": meta.y_meter_start,
                "y_stop": meta.y_meter_stop,
            },
            "num_frames": meta.num_frames,
            "sha256": sha256_bytes(data),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "demos/catalog.json")
    parser.add_argument("--frames-root", type=Path, default=EXTRACTED)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument("--update", action="store_true", help="rewrite the catalog from data")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    with tempfile.TemporaryDirectory() as tmp:
        zips = Path(tmp)
        entries = []
        for demo in catalog.demos:
            entry = from_data(demo, args.frames_root, zips)
            if entry != demo and not args.update:
                raise SystemExit(f"{demo.demo_id}: the catalog differs from the data; --update")
            entries.append(entry)
            print(
                f"{entry.demo_id}: {entry.num_frames} frames, reference "
                f"→ {entry.reference.right} ← {entry.reference.left}, sha256 {entry.sha256[:12]}…"
            )
        catalog = DemoCatalog(version=catalog.version, demos=tuple(entries))
        if args.update:
            args.catalog.write_text(catalog_json(catalog), encoding="utf-8")
            print(f"updated {args.catalog}")
        archive = pack_demos(catalog, zips, REPO_ROOT, args.out)
    print(f"wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
