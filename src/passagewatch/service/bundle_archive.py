"""Release bundles as downloadable archives, verified against the committed release manifest.

Model weights are not in git. Each release's inference bundle (``bundle.json``, the detector
checkpoint and, for ONNX releases, the exported network) is published as
``<release>-bundle.tar.gz`` on the GitHub release. The archive is reproducible (sorted
members, fixed times and owners), so packing the same bundle twice gives the same SHA-256.

A bundle is accepted only if it matches ``releases/manifests/<release>.json``: its
pipeline configuration hash, detector, preprocessing, tracker, counting-policy and
calibration fields, and the SHA-256 of every weight file. Downloading therefore gives
exactly the evaluated release, or an error.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path

from passagewatch.service.bundle import BUNDLE_FILE, load_bundle
from passagewatch.service.release import ReleaseManifest, bundle_identity

ARCHIVE_SUFFIX = "-bundle.tar.gz"
NOTICES = ("LICENSE", "THIRD_PARTY_NOTICES.md")  # shipped with the weights


class BundleMismatchError(ValueError):
    """A bundle that is not the release its manifest describes."""


def archive_name(release: str) -> str:
    return f"{release}{ARCHIVE_SUFFIX}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle_dir: Path, manifest: ReleaseManifest) -> None:
    """Raise ``BundleMismatchError`` unless ``bundle_dir`` is the manifest's release."""
    bundle = load_bundle(bundle_dir)
    problems = []
    if bundle.pipeline_version != manifest.release:
        problems.append(f"pipeline_version {bundle.pipeline_version} != {manifest.release}")
    identity = bundle_identity(bundle)
    for key, want in manifest.bundle.items():
        if identity.get(key) != want:
            problems.append(f"{key}: {identity.get(key)} != {want}")
    detector = bundle.detector
    files: list[tuple[str, str | None]] = [(detector.checkpoint, detector.checkpoint_sha256)]
    if detector.onnx_file is not None:
        files.append((detector.onnx_file, detector.onnx_sha256))
    for name, want_sha in files:
        path = bundle_dir / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif sha256_file(path) != want_sha:
            problems.append(f"{name}: SHA-256 differs from bundle.json")
    if problems:
        raise BundleMismatchError(f"{bundle_dir} is not {manifest.release}: " + "; ".join(problems))


def pack_bundle(
    bundle_dir: Path, manifest: ReleaseManifest, out_dir: Path, notices_dir: Path
) -> Path:
    """Verify ``bundle_dir`` and write ``out_dir/<release>-bundle.tar.gz``; return its path."""
    verify_bundle(bundle_dir, manifest)
    bundle = load_bundle(bundle_dir)
    names = [BUNDLE_FILE, bundle.detector.checkpoint]
    if bundle.detector.onnx_file is not None:
        names.append(bundle.detector.onnx_file)
    members = [(f"{manifest.release}/{n}", bundle_dir / n) for n in names]
    members += [(f"{manifest.release}/{n}", notices_dir / n) for n in NOTICES]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / archive_name(manifest.release)
    write_tar_gz(members, out)
    return out


def write_tar_gz(members: list[tuple[str, Path]], out: Path) -> None:
    """Write ``(arcname, path)`` members as a reproducible ``.tar.gz``: sorted members,
    fixed times, owners and modes, so the same files always give the same SHA-256."""
    with (
        out.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for arcname, path in sorted(members):
            info = tarfile.TarInfo(arcname)
            info.size = path.stat().st_size
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as fh:
                tar.addfile(info, fh)


def unpack_bundle(archive: Path, manifest: ReleaseManifest, bundles_dir: Path) -> Path:
    """Extract a downloaded archive to ``bundles_dir/<release>/`` after verifying it.

    Only regular files directly under ``<release>/`` are accepted. The bundle is verified
    in a temporary directory and moved into place only if it matches the manifest. An
    existing bundle of that release is kept if it matches, and refused otherwise.
    """
    release = manifest.release
    target = bundles_dir / release
    if target.exists():
        verify_bundle(target, manifest)
        return target
    bundles_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=bundles_dir, prefix=f".{release}-") as tmp:
        staging = Path(tmp) / release
        staging.mkdir()
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar.getmembers():
                parent, _, name = member.name.partition("/")
                if parent != release or not name or "/" in name or name in ("..", "."):
                    raise BundleMismatchError(f"unexpected archive member {member.name!r}")
                if not member.isfile():
                    raise BundleMismatchError(f"{member.name!r} is not a regular file")
                source = tar.extractfile(member)
                assert source is not None
                with source, (staging / name).open("wb") as fh:
                    shutil.copyfileobj(source, fh)
        verify_bundle(staging, manifest)
        staging.rename(target)
    return target
