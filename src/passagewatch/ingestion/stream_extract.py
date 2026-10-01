"""Stream a remote tar archive and keep only selected members, verifying the whole archive.

Used when an archive is too large to store: the bytes go straight from HTTP through the
decompressor into the selected files, and the archive itself is never written to disk. The
MD5 and size of the **whole** archive are still checked against the publisher's values.
Selected files are written to a temporary directory that is renamed into place only after
the check passes, so a failed or corrupted stream never leaves a directory that looks
complete. A dropped connection resumes with an HTTP ``Range`` request into the same
decompressor; a server that ignores ``Range`` cannot resume and fails the stream.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from passagewatch.ingestion.download import (
    CHUNK_SIZE,
    USER_AGENT,
    ChecksumError,
    ProgressCallback,
)
from passagewatch.ingestion.extract import EXTRACTED_MARKER, UnsafeArchiveError
from passagewatch.ingestion.sources import ResolvedFile

logger = logging.getLogger(__name__)

# Maps an archive member name to a destination path relative to the output directory,
# or None to skip it.
MemberSelector = Callable[[str], PurePosixPath | None]


class StreamError(ConnectionError):
    """The stream could not be resumed."""


class _VerifyingHttpStream(io.RawIOBase):
    """A read-only stream over an HTTP body that hashes every byte and resumes on errors."""

    def __init__(
        self,
        url: str,
        size: int,
        *,
        timeout: float,
        retries: int,
        backoff_seconds: float,
        progress: ProgressCallback | None,
    ) -> None:
        super().__init__()
        self.url = url
        self.size = size
        self.timeout = timeout
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.progress = progress
        self.position = 0
        self.md5 = hashlib.md5(usedforsecurity=False)
        self._response: Any = None

    def readable(self) -> bool:
        return True

    def _open(self) -> None:
        request = urllib.request.Request(self.url, headers={"User-Agent": USER_AGENT})
        if self.position:
            request.add_header("Range", f"bytes={self.position}-")
        response = urllib.request.urlopen(request, timeout=self.timeout)
        if self.position and response.status != 206:
            response.close()
            raise StreamError(f"server ignored Range at byte {self.position}; cannot resume")
        self._response = response

    def _close_response(self) -> None:
        if self._response is not None:
            self._response.close()
            self._response = None

    def readinto(self, buffer: Any) -> int:
        if self.position >= self.size:
            return 0
        for attempt in range(1, self.retries + 1):
            try:
                if self._response is None:
                    self._open()
                n = int(self._response.readinto(buffer))
                if n == 0:
                    raise ConnectionError(f"connection ended at byte {self.position}")
                break
            except StreamError:
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                self._close_response()
                if attempt == self.retries:
                    raise
                delay = self.backoff_seconds * 2 ** (attempt - 1)
                logger.warning(
                    "stream of %s failed at byte %d (attempt %d/%d): %s; resuming in %.0fs",
                    self.url,
                    self.position,
                    attempt,
                    self.retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
        view = memoryview(buffer)[:n]
        self.md5.update(view)
        self.position += n
        if self.progress is not None:
            self.progress(self.position, self.size)
        return n

    def drain(self) -> None:
        """Read (and hash) whatever the tar reader did not consume, e.g. trailing padding."""
        buffer = bytearray(CHUNK_SIZE)
        while self.readinto(buffer):
            pass

    def close(self) -> None:
        self._close_response()
        super().close()


@dataclass
class StreamResult:
    path: Path
    members_seen: int
    files_written: int
    bytes_written: int
    skipped: Counter[str] = field(default_factory=Counter)


def _safe_destination(name: str, relative: PurePosixPath) -> PurePosixPath:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise UnsafeArchiveError(f"unsafe destination {relative} for member {name!r}")
    return relative


def stream_extract(
    source: ResolvedFile,
    dest: Path,
    select: MemberSelector,
    *,
    description: dict[str, Any] | None = None,
    retries: int = 8,
    timeout: float = 60.0,
    backoff_seconds: float = 2.0,
    progress: ProgressCallback | None = None,
) -> StreamResult:
    """Stream ``source`` and write the members chosen by ``select`` under ``dest``.

    ``description`` (for example, the selection rule) is stored in the completion marker.
    An existing ``dest`` with a marker for the same source and description is reused.
    """
    marker_record = {
        "source": source.ref,
        "source_md5": source.md5,
        "selection": description or {},
    }
    marker = dest / EXTRACTED_MARKER
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if {k: recorded.get(k) for k in marker_record} == marker_record:
            logger.info("already extracted: %s", dest)
            return StreamResult(
                path=dest,
                members_seen=recorded["members_seen"],
                files_written=recorded["files_written"],
                bytes_written=recorded["bytes_written"],
                skipped=Counter(recorded["skipped"]),
            )
        raise RuntimeError(f"{dest} holds a different extraction; remove it first")
    if dest.exists():
        raise RuntimeError(f"{dest} exists without a completion marker; remove it first")

    tmp = dest.with_name(f".{dest.name}.partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    stream = _VerifyingHttpStream(
        source.url,
        source.size,
        timeout=timeout,
        retries=retries,
        backoff_seconds=backoff_seconds,
        progress=progress,
    )
    result = StreamResult(path=dest, members_seen=0, files_written=0, bytes_written=0)
    try:
        reader = io.BufferedReader(stream, buffer_size=CHUNK_SIZE)
        with tarfile.open(fileobj=reader, mode="r|*") as archive:
            for member in archive:
                result.members_seen += 1
                if member.isdir():
                    continue
                relative = select(member.name)
                if relative is None:
                    result.skipped["not_selected"] += 1
                    continue
                if not member.isfile():
                    raise UnsafeArchiveError(f"selected member is not a file: {member.name!r}")
                target = tmp / _safe_destination(member.name, relative)
                if target.exists():
                    raise RuntimeError(f"two members map to {relative}: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                assert extracted is not None
                with target.open("wb") as out:
                    shutil.copyfileobj(extracted, out, CHUNK_SIZE)
                result.files_written += 1
                result.bytes_written += member.size
        # The tar reader stops at the end-of-archive marker; hash the rest of the stream.
        reader.read()
        stream.drain()

        if stream.position != source.size:
            raise ChecksumError(
                f"{source.ref}: expected {source.size} bytes, streamed {stream.position}"
            )
        actual = stream.md5.hexdigest()
        if actual != source.md5:
            raise ChecksumError(f"{source.ref}: expected md5 {source.md5}, got {actual}")

        (tmp / EXTRACTED_MARKER).write_text(
            json.dumps(
                {
                    **marker_record,
                    "members_seen": result.members_seen,
                    "files_written": result.files_written,
                    "bytes_written": result.bytes_written,
                    "skipped": dict(result.skipped),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        stream.close()
    return result
