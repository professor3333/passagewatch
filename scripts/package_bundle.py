"""Package a release's inference bundle for publication on its GitHub release.

Verifies ``bundles/<release>/`` against ``releases/manifests/<release>.json``, then writes
``dist/<release>-bundle.tar.gz`` (reproducible) and its ``.sha256``. Publish both with:

    gh release upload <tag> dist/<release>-bundle.tar.gz dist/<release>-bundle.tar.gz.sha256

Example:
    uv run python scripts/package_bundle.py --release passagewatch-0.3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from passagewatch.service.bundle_archive import pack_bundle, sha256_file
from passagewatch.service.release import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", required=True)
    parser.add_argument("--bundles", type=Path, default=REPO_ROOT / "bundles")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "dist")
    args = parser.parse_args(argv)

    manifest = read_manifest(REPO_ROOT / "releases/manifests" / f"{args.release}.json")
    archive = pack_bundle(args.bundles / args.release, manifest, args.out, REPO_ROOT)
    digest = sha256_file(archive)
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"{archive}: {archive.stat().st_size / 1e6:.1f} MB, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
