"""Resumable, checksum-verified downloads into immutable raw storage.

A file is written to ``<name>.part`` and promoted to its final name only after its size
and MD5 match the publisher's values. A verified file is never overwritten. Interrupted
downloads resume with an HTTP ``Range`` request; each attempt re-requests the publisher
URL, because CaltechDATA redirects to short-lived signed storage URLs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from passagewatch.ingestion.sources import ResolvedFile

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1 << 20
VERIFIED_SUFFIX = ".verified.json"
# CaltechDATA rejects Python's default "Python-urllib" user agent with 403 Forbidden.
USER_AGENT = "passagewatch-data/0.1 (+https://github.com/professor3333/passagewatch)"

ProgressCallback = Callable[[int, int], None]


class ChecksumError(RuntimeError):
    """The downloaded bytes do not match the publisher's size or checksum."""


class IncompleteDownloadError(ConnectionError):
    """The connection ended before all bytes arrived; the download can resume."""


# Client errors that retrying cannot fix.
_PERMANENT_HTTP_STATUSES = frozenset(range(400, 500)) - {408, 429}


def md5_of(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def destination_for(source: ResolvedFile, raw_dir: Path) -> Path:
    return raw_dir / source.record / source.key


def _marker_path(path: Path) -> Path:
    return path.with_name(path.name + VERIFIED_SUFFIX)


def is_verified(source: ResolvedFile, path: Path) -> bool:
    """True if ``path`` holds ``source`` and was verified without changing since."""
    marker = _marker_path(path)
    if not path.is_file() or not marker.is_file():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    stat = path.stat()
    return bool(
        record.get("md5") == source.md5
        and record.get("size") == source.size == stat.st_size
        and record.get("mtime_ns") == stat.st_mtime_ns
    )


def _write_marker(source: ResolvedFile, path: Path) -> None:
    stat = path.stat()
    marker = {
        "ref": source.ref,
        "url": source.url,
        "size": stat.st_size,
        "md5": source.md5,
        "mtime_ns": stat.st_mtime_ns,
        "verified_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _marker_path(path).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _fetch_into(
    url: str,
    part: Path,
    expected_size: int,
    timeout: float,
    progress: ProgressCallback | None,
) -> None:
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_size:
        part.unlink()
        offset = 0
    if offset == expected_size:
        return

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        if offset and status == 200:
            # The server ignored the Range header; start over rather than append.
            logger.warning("server ignored Range for %s; restarting download", url)
            offset = 0
        elif status not in (200, 206):
            raise urllib.error.HTTPError(url, status, "unexpected status", response.headers, None)
        mode = "ab" if offset else "wb"
        written = offset
        with part.open(mode) as fh:
            while chunk := response.read(CHUNK_SIZE):
                fh.write(chunk)
                written += len(chunk)
                if progress is not None:
                    progress(written, expected_size)
    if written < expected_size:
        raise IncompleteDownloadError(f"received {written} of {expected_size} bytes from {url}")


def download(
    source: ResolvedFile,
    raw_dir: Path,
    *,
    retries: int = 5,
    timeout: float = 60.0,
    backoff_seconds: float = 2.0,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download ``source`` into ``raw_dir`` and return the verified path."""
    dest = destination_for(source, raw_dir)
    if is_verified(source, dest):
        logger.info("already verified: %s", source.ref)
        return dest
    if dest.exists():
        # A final file without a matching marker: verify it rather than trust or replace it.
        if dest.stat().st_size == source.size and md5_of(dest) == source.md5:
            _write_marker(source, dest)
            return dest
        raise ChecksumError(
            f"{dest} exists but does not match {source.ref}; move it aside and retry"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            _fetch_into(source.url, part, source.size, timeout, progress)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            permanent = (
                isinstance(exc, urllib.error.HTTPError) and exc.code in _PERMANENT_HTTP_STATUSES
            )
            if permanent or attempt == retries:
                raise
            delay = backoff_seconds * 2 ** (attempt - 1)
            logger.warning(
                "download of %s failed (attempt %d/%d): %s; retrying in %.0fs",
                source.ref,
                attempt,
                retries,
                exc,
                delay,
            )
            time.sleep(delay)

    size = part.stat().st_size
    if size != source.size:
        raise ChecksumError(f"{source.ref}: expected {source.size} bytes, got {size}")
    actual = md5_of(part)
    if actual != source.md5:
        part.unlink()
        raise ChecksumError(f"{source.ref}: expected md5 {source.md5}, got {actual}")
    part.rename(dest)
    _write_marker(source, dest)
    logger.info("downloaded and verified: %s", source.ref)
    return dest
