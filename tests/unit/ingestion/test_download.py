from __future__ import annotations

import hashlib
import urllib.error
from pathlib import Path

import pytest

from passagewatch.ingestion.download import (
    USER_AGENT,
    ChecksumError,
    destination_for,
    download,
    is_verified,
)
from passagewatch.ingestion.sources import ResolvedFile

from .conftest import FakeServer

BODY = bytes(range(256)) * 400  # 102,400 bytes


def _source(server: FakeServer, key: str = "data.bin", body: bytes = BODY) -> ResolvedFile:
    return ResolvedFile(
        record="rec-1",
        key=key,
        size=len(body),
        md5=hashlib.md5(body, usedforsecurity=False).hexdigest(),
        url=f"{server.base_url}/redirect/{key}?download=1",
    )


def _download(source: ResolvedFile, raw_dir: Path) -> Path:
    return download(source, raw_dir, retries=3, timeout=5, backoff_seconds=0)


def test_downloads_through_redirect_and_verifies(fake_server: FakeServer, tmp_path: Path) -> None:
    fake_server.behavior.files["data.bin"] = BODY
    source = _source(fake_server)

    path = _download(source, tmp_path)

    assert path == tmp_path / "rec-1" / "data.bin"
    assert path.read_bytes() == BODY
    assert is_verified(source, path)
    assert not path.with_name("data.bin.part").exists()
    # CaltechDATA answers 403 to the default Python user agent.
    assert set(fake_server.behavior.user_agents) == {USER_AGENT}


def test_verified_file_is_not_downloaded_again(fake_server: FakeServer, tmp_path: Path) -> None:
    fake_server.behavior.files["data.bin"] = BODY
    source = _source(fake_server)
    _download(source, tmp_path)
    requests_after_first = len(fake_server.behavior.requests)

    _download(source, tmp_path)

    assert len(fake_server.behavior.requests) == requests_after_first


def test_resumes_partial_download_with_range(fake_server: FakeServer, tmp_path: Path) -> None:
    fake_server.behavior.files["data.bin"] = BODY
    source = _source(fake_server)
    part = destination_for(source, tmp_path).with_name("data.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(BODY[:40_000])

    path = _download(source, tmp_path)

    assert path.read_bytes() == BODY
    ranges = [r for p, r in fake_server.behavior.requests if p.startswith("/files/")]
    assert ranges == ["bytes=40000-"]


def test_dropped_connection_is_resumed(fake_server: FakeServer, tmp_path: Path) -> None:
    behavior = fake_server.behavior
    behavior.files["data.bin"] = BODY
    behavior.truncate_after = 30_000
    behavior.truncate_times = 1

    path = _download(_source(fake_server), tmp_path)

    assert path.read_bytes() == BODY
    ranges = [r for p, r in behavior.requests if p.startswith("/files/")]
    assert ranges == [None, "bytes=30000-"]


def test_server_ignoring_range_restarts_instead_of_appending(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.behavior.files["data.bin"] = BODY
    fake_server.behavior.honor_range = False
    source = _source(fake_server)
    part = destination_for(source, tmp_path).with_name("data.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(BODY[:40_000])

    path = _download(source, tmp_path)

    assert path.read_bytes() == BODY


def test_checksum_mismatch_is_rejected_and_not_promoted(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.behavior.files["data.bin"] = BODY
    wrong = _source(fake_server).model_copy(update={"md5": "0" * 32})

    with pytest.raises(ChecksumError, match="expected md5"):
        _download(wrong, tmp_path)

    dest = destination_for(wrong, tmp_path)
    assert not dest.exists()
    assert not dest.with_name("data.bin.part").exists()


def test_existing_unverified_file_that_mismatches_is_not_overwritten(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.behavior.files["data.bin"] = BODY
    source = _source(fake_server)
    dest = destination_for(source, tmp_path)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"something else")

    with pytest.raises(ChecksumError, match="does not match"):
        _download(source, tmp_path)

    assert dest.read_bytes() == b"something else"


def test_existing_matching_file_is_adopted_without_download(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    source = _source(fake_server)
    dest = destination_for(source, tmp_path)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(BODY)

    _download(source, tmp_path)

    assert is_verified(source, dest)
    assert fake_server.behavior.requests == []


def test_modified_verified_file_is_no_longer_trusted(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    fake_server.behavior.files["data.bin"] = BODY
    source = _source(fake_server)
    path = _download(source, tmp_path)

    path.write_bytes(BODY[:-1] + b"\x00")

    assert not is_verified(source, path)


def test_missing_file_fails_without_retrying(fake_server: FakeServer, tmp_path: Path) -> None:
    source = _source(fake_server, key="absent.bin")

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _download(source, tmp_path)

    assert excinfo.value.code == 404
    file_requests = [p for p, _ in fake_server.behavior.requests if p.startswith("/files/")]
    assert len(file_requests) == 1
