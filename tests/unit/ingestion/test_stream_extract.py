from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from passagewatch.ingestion.download import ChecksumError
from passagewatch.ingestion.extract import EXTRACTED_MARKER
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.sources import ResolvedFile
from passagewatch.ingestion.stream_extract import StreamError, stream_extract
from passagewatch.ingestion.subsets import (
    CfcFrameSelector,
    Include,
    SubsetConfig,
    select_clips,
)

from .conftest import FakeServer

KEEP = "cam_2018-05-26_120000_0_3"
DROP = "cam_2018-05-27_120000_0_3"


def _dir() -> tarfile.TarInfo:
    info = tarfile.TarInfo("kenai/")
    info.type = tarfile.DIRTYPE
    return info


def make_archive(members: dict[str, bytes]) -> bytes:
    """A gzip-compressed tar (like CFC's kenai.tar) with a directory entry first."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        tf.addfile(_dir())
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def frames(clip: str, n: int = 3, size: int = 40_000) -> dict[str, bytes]:
    rng = np.random.default_rng(len(clip))
    return {f"kenai/{clip}_{i}.jpg": rng.bytes(size) for i in range(n)}


ARCHIVE_MEMBERS = {**frames(KEEP), **frames(DROP), "kenai/stray.txt": b"x"}


def source_for(server: FakeServer, body: bytes, md5: str | None = None) -> ResolvedFile:
    server.behavior.files["kenai.tar"] = body
    return ResolvedFile(
        record="rec",
        key="kenai.tar",
        size=len(body),
        md5=md5 or hashlib.md5(body, usedforsecurity=False).hexdigest(),
        url=f"{server.base_url}/redirect/kenai.tar?download=1",
    )


def selector() -> CfcFrameSelector:
    return CfcFrameSelector("kenai/", {KEEP: "kenai-train"}, {KEEP, DROP})


def extract(source: ResolvedFile, dest: Path, sel: CfcFrameSelector | None = None) -> object:
    return stream_extract(
        source,
        dest,
        sel or selector(),
        description={"name": "t"},
        retries=4,
        timeout=5,
        backoff_seconds=0,
    )


def test_keeps_only_selected_frames_and_verifies_the_archive(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    source = source_for(fake_server, make_archive(ARCHIVE_MEMBERS))
    sel = selector()

    result = extract(source, tmp_path / "subset", sel)

    kept = sorted(
        p.relative_to(tmp_path / "subset").as_posix() for p in (tmp_path / "subset").rglob("*.jpg")
    )
    assert kept == [f"kenai-train/{KEEP}/{i}.jpg" for i in range(3)]
    assert (tmp_path / "subset" / "kenai-train" / KEEP / "1.jpg").read_bytes() == (
        ARCHIVE_MEMBERS[f"kenai/{KEEP}_1.jpg"]
    )
    assert sel.skipped == {"not_selected": 3, "unexpected_name": 1}
    assert sel.unknown_examples == ["kenai/stray.txt"]
    marker = json.loads((tmp_path / "subset" / EXTRACTED_MARKER).read_text())
    assert marker["source_md5"] == source.md5 and marker["files_written"] == 3
    assert result.files_written == 3  # type: ignore[attr-defined]


def test_resumes_dropped_connections_into_the_same_stream(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    source = source_for(fake_server, make_archive(ARCHIVE_MEMBERS))
    fake_server.behavior.truncate_after = 30_000
    fake_server.behavior.truncate_times = 3

    extract(source, tmp_path / "subset")

    ranges = [r for path, r in fake_server.behavior.requests if r and path.startswith("/files/")]
    assert len(ranges) == 3
    assert len(list((tmp_path / "subset").rglob("*.jpg"))) == 3


def test_checksum_mismatch_leaves_nothing_behind(fake_server: FakeServer, tmp_path: Path) -> None:
    source = source_for(fake_server, make_archive(ARCHIVE_MEMBERS), md5="0" * 32)

    with pytest.raises(ChecksumError, match="expected md5"):
        extract(source, tmp_path / "subset")

    assert list(tmp_path.iterdir()) == []


def test_server_ignoring_range_cannot_resume(fake_server: FakeServer, tmp_path: Path) -> None:
    source = source_for(fake_server, make_archive(ARCHIVE_MEMBERS))
    fake_server.behavior.truncate_after = 30_000
    fake_server.behavior.truncate_times = 1
    fake_server.behavior.honor_range = False

    with pytest.raises(StreamError, match="ignored Range"):
        extract(source, tmp_path / "subset")
    assert list(tmp_path.iterdir()) == []


def test_completed_extraction_is_reused_and_others_refused(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    source = source_for(fake_server, make_archive(ARCHIVE_MEMBERS))
    extract(source, tmp_path / "subset")
    requests = len(fake_server.behavior.requests)

    extract(source, tmp_path / "subset")
    assert len(fake_server.behavior.requests) == requests

    with pytest.raises(RuntimeError, match="different extraction"):
        stream_extract(source, tmp_path / "subset", selector(), description={"name": "other"})


def test_unsafe_destination_is_rejected(fake_server: FakeServer, tmp_path: Path) -> None:
    source = source_for(fake_server, make_archive(ARCHIVE_MEMBERS))

    with pytest.raises(Exception, match="unsafe destination"):
        stream_extract(
            source,
            tmp_path / "subset",
            lambda name: PurePosixPath("../escape.jpg") if name.endswith(".jpg") else None,
            retries=1,
            timeout=5,
        )
    assert not (tmp_path / "escape.jpg").exists()


def meta(name: str) -> ClipMetadata:
    return ClipMetadata(
        clip_name=name,
        num_frames=3,
        framerate=10.0,
        width=10,
        height=10,
        x_meter_start=0,
        x_meter_stop=1,
        y_meter_start=1,
        y_meter_stop=0,
    )


def test_subset_selects_whole_days_and_whole_locations() -> None:
    config = SubsetConfig(
        name="dev-v1",
        archive="rec/kenai.tar",
        member_prefix="kenai/",
        include=(
            Include(location="kenai-train", days=(dt.date(2018, 5, 26),)),
            Include(location="kenai-val"),
        ),
    )
    val = "v_2018-06-03_120000_0_3"
    metadata = {"kenai-train": {KEEP: meta(KEEP), DROP: meta(DROP)}, "kenai-val": {val: meta(val)}}

    assert select_clips(config, metadata) == {KEEP: "kenai-train", val: "kenai-val"}

    missing_day = config.model_copy(
        update={"include": (Include(location="kenai-train", days=(dt.date(2018, 7, 1),)),)}
    )
    with pytest.raises(ValueError, match="no clips on"):
        select_clips(missing_day, metadata)


def test_repository_subset_config_is_valid() -> None:
    from passagewatch.ingestion.subsets import load_subset_config

    path = Path(__file__).resolve().parents[3] / "configs/data/kenai_subset.yaml"
    config = load_subset_config(path)

    assert config.archive == "g945x-41103/kenai.tar"
    assert {i.location for i in config.include} == {"kenai-train", "kenai-val"}
