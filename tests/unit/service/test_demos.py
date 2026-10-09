from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from passagewatch.service.demos import (
    DemoCatalog,
    DemoMismatchError,
    frames_zip,
    pack_demos,
    unpack_demos,
)


def frames(tmp_path: Path, n: int, seed: int = 0) -> list[Path]:
    paths = []
    for i in range(n):
        path = tmp_path / f"src-{seed}" / f"{i}.jpg"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(bytes([seed, i]) * 50)
        paths.append(path)
    return paths


def entry(demo_id: str, sha256: str) -> dict[str, object]:
    return {
        "demo_id": demo_id,
        "kind": "clear",
        "title": demo_id,
        "summary": "",
        "source": {
            "dataset": "Caltech Fish Counting (CFC) v1.1",
            "location": "kenai-train",
            "clip_name": demo_id,
            "split_note": "",
        },
        "reference": {"right": 1, "left": 0},
        "framerate": 10,
        "meters": {"x_start": -1, "x_stop": 1, "y_start": 3, "y_stop": 0},
        "num_frames": 3,
        "sha256": sha256,
    }


def test_frames_zip_is_reproducible_and_numbered_from_zero(tmp_path: Path) -> None:
    paths = frames(tmp_path, 3)
    data = frames_zip(paths)

    assert frames_zip(paths) == data
    names = zipfile.ZipFile(__import__("io").BytesIO(data)).namelist()
    assert names == ["0.jpg", "1.jpg", "2.jpg"]


def test_pack_and_unpack_verify_every_zip(tmp_path: Path) -> None:
    zips, notices = tmp_path / "zips", tmp_path / "notices"
    zips.mkdir()
    notices.mkdir()
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        (notices / name).write_text(name)
    shas = {}
    for seed, demo_id in enumerate(("clear", "difficult")):
        data = frames_zip(frames(tmp_path, 3, seed))
        (zips / f"{demo_id}.zip").write_bytes(data)
        shas[demo_id] = hashlib.sha256(data).hexdigest()
    catalog = DemoCatalog.model_validate(
        {"version": "demos-v1", "demos": [entry(k, v) for k, v in shas.items()]}
    )

    archive = pack_demos(catalog, zips, notices, tmp_path / "dist")
    assert archive.name == "passagewatch-demos-v1.tar.gz"
    first = archive.read_bytes()
    assert pack_demos(catalog, zips, notices, tmp_path / "dist").read_bytes() == first

    found = unpack_demos(archive, catalog, tmp_path / "out")
    assert sorted(found) == ["clear", "difficult"]

    wrong = catalog.model_copy(
        update={
            "demos": (
                catalog.demos[0],
                catalog.demos[0].model_copy(update={"demo_id": "other", "sha256": "0" * 64}),
            )
        }
    )
    with pytest.raises(DemoMismatchError):
        unpack_demos(archive, wrong, tmp_path / "out2")


def test_catalog_entries_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        DemoCatalog.model_validate(
            {"version": "demos-v1", "demos": [entry("one", "1" * 64), entry("two", "1" * 64)]}
        )
