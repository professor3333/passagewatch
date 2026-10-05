"""Download a release's inference bundle and verify it against the committed manifest.

The bundle (``bundle.json``, detector weights, ONNX network) is published on the GitHub
release as ``<release>-bundle.tar.gz``. It is unpacked into ``bundles/<release>/`` only if
its pipeline configuration and every weight file's SHA-256 match
``releases/manifests/<release>.json``; ``--activate`` then points ``bundles/active`` at it.

Examples:
    uv run python scripts/fetch_bundle.py --release passagewatch-0.3.0 --activate
    # the previous release (the rollback target), published with 0.3.0:
    uv run python scripts/fetch_bundle.py --release passagewatch-0.2.0 --tag passagewatch-0.3.0
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

from passagewatch.service.bundle import activate
from passagewatch.service.bundle_archive import archive_name, sha256_file, unpack_bundle
from passagewatch.service.release import read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = "https://github.com/professor3333/passagewatch/releases/download"


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "passagewatch-fetch-bundle"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release", required=True, help="the bundle's release, e.g. passagewatch-0.3.0"
    )
    parser.add_argument("--tag", help="the GitHub release it is attached to (default: --release)")
    parser.add_argument("--archive", type=Path, help="use a downloaded archive instead")
    parser.add_argument("--bundles", type=Path, default=REPO_ROOT / "bundles")
    parser.add_argument("--activate", action="store_true", help="point bundles/active at it")
    args = parser.parse_args(argv)

    manifest = read_manifest(REPO_ROOT / "releases/manifests" / f"{args.release}.json")
    with tempfile.TemporaryDirectory() as tmp:
        archive = args.archive
        if archive is None:
            url = f"{DOWNLOADS}/{args.tag or args.release}/{archive_name(args.release)}"
            print(f"downloading {url}")
            archive = Path(tmp) / archive_name(args.release)
            download(url, archive)
        print(f"archive sha256 {sha256_file(archive)}")
        target = unpack_bundle(archive, manifest, args.bundles)
    print(f"{target} matches releases/manifests/{args.release}.json")
    if args.activate:
        activate(args.bundles, args.release)
        print(f"bundles/active -> {args.release}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
