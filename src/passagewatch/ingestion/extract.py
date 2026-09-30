"""Safe, atomic archive extraction.

Archives are unpacked into a temporary sibling directory and renamed into place only when
extraction finishes, so a crash never leaves a half-extracted directory that looks complete.
Members that would escape the destination (absolute paths, ``..``, links outside the tree,
device files) are rejected.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

EXTRACTED_MARKER = ".extracted.json"
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")


class UnsafeArchiveError(ValueError):
    """An archive member would be written outside the extraction directory."""


def archive_stem(path: Path) -> str:
    name = path.name
    for suffix in ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"not a supported archive: {path}")


def is_archive(path: Path) -> bool:
    return path.name.endswith(ARCHIVE_SUFFIXES)


def _check_zip_member(name: str) -> None:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        raise UnsafeArchiveError(f"unsafe zip member path: {name!r}")


def _extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            _check_zip_member(name)
        zf.extractall(target)


def _extract_tar(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as tf:
        try:
            # The "data" filter rejects absolute paths, parent traversal, links that leave
            # the tree, and special files.
            tf.extractall(target, filter="data")
        except tarfile.FilterError as exc:
            raise UnsafeArchiveError(str(exc)) from exc


def extract(archive: Path, extract_root: Path, *, source_md5: str) -> Path:
    """Extract ``archive`` to ``extract_root/<archive stem>`` and return that directory.

    An existing extraction is reused when its marker records the same source checksum.
    """
    dest = extract_root / archive_stem(archive)
    marker = dest / EXTRACTED_MARKER
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("source_md5") == source_md5:
            return dest
        raise RuntimeError(f"{dest} was extracted from a different archive; remove it first")
    if dest.exists():
        raise RuntimeError(f"{dest} exists without a completion marker; remove it first")

    extract_root.mkdir(parents=True, exist_ok=True)
    tmp = extract_root / f".{dest.name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        if archive.name.endswith(".zip"):
            _extract_zip(archive, tmp)
        else:
            _extract_tar(archive, tmp)
        (tmp / EXTRACTED_MARKER).write_text(
            json.dumps({"archive": archive.name, "source_md5": source_md5}, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest
