from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from passagewatch.service.media import (
    UploadTooLargeError,
    iter_frames,
    probe,
    save_upload,
)
from passagewatch.service.settings import ServiceSettings


def write_zip(path: Path, frames: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in frames.items():
            zf.writestr(name, data)
    return path


def png(value: int) -> bytes:
    return bytes(cv2.imencode(".png", np.full((6, 4), value, dtype=np.uint8))[1])


def test_frames_are_read_in_numeric_order(tmp_path: Path) -> None:
    # Lexical order would put 10 before 2.
    path = write_zip(
        tmp_path / "c.zip", {f"{i}.png": png(i) for i in (10, 2, 0, 1, 3, 4, 5, 6, 7, 8, 9)}
    )

    values = [int(f[0, 0]) for f in iter_frames(path, "frames")]

    assert values == list(range(11))
    assert probe(path, max_frames=100).num_frames == 11


def test_a_corrupt_frame_fails_with_its_number(tmp_path: Path) -> None:
    path = write_zip(tmp_path / "c.zip", {"0.png": png(1), "1.png": b"garbage", "2.png": png(3)})

    with pytest.raises(ValueError, match="frame 1"):
        list(iter_frames(path, "frames"))


def test_video_frames_are_grayscale(tmp_path: Path) -> None:
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 5.0, (32, 48))
    for i in range(4):
        writer.write(np.full((48, 32, 3), 30 * i + 20, dtype=np.uint8))
    writer.release()

    frames = list(iter_frames(path, "video"))

    assert len(frames) == 4 and all(f.shape == (48, 32) and f.dtype == np.uint8 for f in frames)


def test_save_upload_hashes_and_enforces_the_limit(tmp_path: Path) -> None:
    saved = save_upload(io.BytesIO(b"abc" * 10), tmp_path / "u.bin", max_bytes=100)
    assert saved.size == 30 and len(saved.sha256) == 64

    with pytest.raises(UploadTooLargeError):
        save_upload(io.BytesIO(b"x" * 101), tmp_path / "big.bin", max_bytes=100)
    assert not list(tmp_path.glob("big.bin*"))


def test_settings_come_from_the_environment() -> None:
    settings = ServiceSettings.from_env(
        {"PASSAGEWATCH_DATA_DIR": "/srv/pw", "PASSAGEWATCH_MAX_QUEUE": "5", "UNRELATED": "x"}
    )

    assert settings.db_path == Path("/srv/pw/service.db") and settings.max_queue == 5
    assert settings.upload_retention_hours == 24.0
    assert settings.torch_threads is None
    assert ServiceSettings.from_env({"PASSAGEWATCH_TORCH_THREADS": "4"}).torch_threads == 4
