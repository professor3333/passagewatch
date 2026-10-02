"""Uploaded recordings: streamed storage with a size limit, validation, and frame reading.

Two formats are accepted:

- **frames:** a ZIP of frames named ``0.jpg``/``0.png`` ... ``N-1``, numbered from 0 without
  gaps (the CFC layout), optionally inside one folder. macOS's ``__MACOSX/`` entries and
  ``.DS_Store`` files are ignored; anything else is rejected.
- **video:** any container OpenCV can decode (MP4, AVI, MOV, ...).

Uploads are streamed to disk and hashed as they arrive, and rejected as soon as they exceed
the size limit. Validation reads the frame list and decodes the first frame; a frame that
turns out to be corrupt later fails the job with that frame's number, instead of being
skipped. :func:`iter_frames` reads frames one at a time, in order and in grayscale, so
memory stays bounded whatever the recording's length.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import cv2
import numpy as np
from numpy.typing import NDArray

CHUNK = 1 << 20
_FRAME_NAME = re.compile(r"^(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


class UploadRejectedError(ValueError):
    """The upload is not an acceptable recording (reported as 422)."""


class UploadTooLargeError(UploadRejectedError):
    """The upload exceeds the size limit (reported as 413)."""


@dataclass(frozen=True)
class SavedUpload:
    path: Path
    size: int
    sha256: str


def save_upload(stream: BinaryIO, dest: Path, *, max_bytes: int) -> SavedUpload:
    """Copy ``stream`` to ``dest`` while hashing it; refuse more than ``max_bytes``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with tmp.open("wb") as out:
            while chunk := stream.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLargeError(f"upload exceeds the {max_bytes}-byte limit")
                digest.update(chunk)
                out.write(chunk)
        if size == 0:
            raise UploadRejectedError("the upload is empty")
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return SavedUpload(path=dest, size=size, sha256=digest.hexdigest())


@dataclass(frozen=True)
class MediaInfo:
    kind: str  # "frames" or "video"
    num_frames: int
    width: int
    height: int
    framerate: float | None  # from the video container; None for frames


def _ignored(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return name.endswith("/") or parts[0] == "__MACOSX" or parts[-1] == ".DS_Store"


def frame_members(archive: zipfile.ZipFile) -> list[str]:
    """Frame member names in frame order; raises if the layout is not ``0..N-1``."""
    by_index: dict[int, str] = {}
    folders = set()
    for name in archive.namelist():
        if _ignored(name):
            continue
        path = PurePosixPath(name)
        if len(path.parts) > 2 or path.is_absolute() or ".." in path.parts:
            raise UploadRejectedError(f"unexpected path in the ZIP: {name!r}")
        match = _FRAME_NAME.match(path.name)
        if match is None:
            raise UploadRejectedError(f"not a frame file (expected <n>.jpg or <n>.png): {name!r}")
        index = int(match.group(1))
        if index in by_index:
            raise UploadRejectedError(f"frame {index} appears twice")
        by_index[index] = name
        folders.add(path.parts[0] if len(path.parts) == 2 else "")
    if not by_index:
        raise UploadRejectedError("the ZIP contains no frames")
    if len(folders) > 1:
        raise UploadRejectedError("frames must be in one folder")
    missing = sorted(set(range(len(by_index))) - set(by_index))
    if missing:
        raise UploadRejectedError(
            f"frames must be numbered 0..N-1 without gaps; missing e.g. {missing[:5]}"
        )
    return [by_index[i] for i in range(len(by_index))]


def _decode(data: bytes, name: str) -> NDArray[np.uint8]:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise UploadRejectedError(f"cannot decode frame {name!r}")
    return np.asarray(image, dtype=np.uint8)


def probe(path: Path, *, max_frames: int) -> MediaInfo:
    """Identify and validate a saved upload."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = frame_members(archive)
            if len(members) > max_frames:
                raise UploadRejectedError(f"{len(members)} frames exceed the limit of {max_frames}")
            first = _decode(archive.read(members[0]), members[0])
        return MediaInfo("frames", len(members), first.shape[1], first.shape[0], None)

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise UploadRejectedError("not a ZIP of frames or a readable video")
        ok, frame = capture.read()
        if not ok:
            raise UploadRejectedError("the video has no readable frames")
        count = 1
        while capture.grab():
            count += 1
            if count > max_frames:
                raise UploadRejectedError(f"the video exceeds the limit of {max_frames} frames")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return MediaInfo(
        "video", count, int(frame.shape[1]), int(frame.shape[0]), fps if fps > 0 else None
    )


def iter_frames(path: Path, kind: str) -> Iterator[NDArray[np.uint8]]:
    """Grayscale frames in order, one at a time. Raises on an undecodable frame."""
    if kind == "frames":
        with zipfile.ZipFile(path) as archive:
            for i, name in enumerate(frame_members(archive)):
                try:
                    yield _decode(archive.read(name), name)
                except UploadRejectedError:
                    raise ValueError(f"frame {i} ({name}) cannot be decoded") from None
        return
    capture = cv2.VideoCapture(str(path))
    try:
        i = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield np.asarray(gray, dtype=np.uint8)
            i += 1
    finally:
        capture.release()


def read_frame(path: Path, kind: str, index: int) -> tuple[bytes, str]:
    """One frame as image bytes and its media type (for display, not for analysis)."""
    if kind == "frames":
        with zipfile.ZipFile(path) as archive:
            members = frame_members(archive)
            if not 0 <= index < len(members):
                raise IndexError(f"frame {index} is outside [0, {len(members)})")
            name = members[index]
            media_type = "image/png" if name.lower().endswith(".png") else "image/jpeg"
            return archive.read(name), media_type
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 <= index < max(count, 1):
            raise IndexError(f"frame {index} is outside [0, {count})")
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise IndexError(f"frame {index} cannot be read")
    return bytes(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]), "image/jpeg"
