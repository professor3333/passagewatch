from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from passagewatch.ingestion.extract import (
    EXTRACTED_MARKER,
    UnsafeArchiveError,
    archive_stem,
    extract,
)


def _make_tar(path: Path, members: dict[str, bytes], mode: str = "w:gz") -> Path:
    with tarfile.open(path, mode) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize(
    ("name", "stem"),
    [
        ("tiny_dataset.tar.gz", "tiny_dataset"),
        ("kenai.tar", "kenai"),
        ("file_lists_v1.1.zip", "file_lists_v1.1"),
    ],
)
def test_archive_stem(name: str, stem: str) -> None:
    assert archive_stem(Path(name)) == stem


def test_extracts_tar_and_writes_marker(tmp_path: Path) -> None:
    archive = _make_tar(tmp_path / "a.tar.gz", {"x/gt.txt": b"1,1\n", "x/0.jpg": b"\xff"})

    dest = extract(archive, tmp_path / "out", source_md5="m1")

    assert dest == tmp_path / "out" / "a"
    assert (dest / "x/gt.txt").read_bytes() == b"1,1\n"
    assert json.loads((dest / EXTRACTED_MARKER).read_text())["source_md5"] == "m1"
    assert not (tmp_path / "out" / ".a.partial").exists()


def test_reuses_completed_extraction(tmp_path: Path) -> None:
    archive = _make_tar(tmp_path / "a.tar.gz", {"f.txt": b"v1"})
    dest = extract(archive, tmp_path / "out", source_md5="m1")
    (dest / "f.txt").write_bytes(b"kept")

    assert extract(archive, tmp_path / "out", source_md5="m1") == dest
    assert (dest / "f.txt").read_bytes() == b"kept"


def test_refuses_extraction_from_a_different_archive(tmp_path: Path) -> None:
    archive = _make_tar(tmp_path / "a.tar.gz", {"f.txt": b"v1"})
    extract(archive, tmp_path / "out", source_md5="m1")

    with pytest.raises(RuntimeError, match="different archive"):
        extract(archive, tmp_path / "out", source_md5="m2")


def test_rejects_tar_path_traversal_and_leaves_nothing(tmp_path: Path) -> None:
    archive = _make_tar(tmp_path / "evil.tar", {"../escape.txt": b"x"}, mode="w")

    with pytest.raises(UnsafeArchiveError):
        extract(archive, tmp_path / "out", source_md5="m")

    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "out" / "evil").exists()
    assert not (tmp_path / "out" / ".evil.partial").exists()


def test_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", b"x")

    with pytest.raises(UnsafeArchiveError):
        extract(archive, tmp_path / "out", source_md5="m")

    assert not (tmp_path / "escape.txt").exists()


def test_extracts_zip(tmp_path: Path) -> None:
    archive = tmp_path / "lists.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("lists/train.txt", b"a\nb\n")

    dest = extract(archive, tmp_path / "out", source_md5="m")

    assert (dest / "lists/train.txt").read_bytes() == b"a\nb\n"
