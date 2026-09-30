"""MOTChallenge-format annotations and their conversion to the internal convention.

CFC ``gt.txt`` rows are ``frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z``,
with **1-based** frames and box coordinates. This module is the only place that converts
them; everything downstream uses :class:`BoxAnnotations` in the internal convention
defined in ``docs/counting_policy.md`` §1:

- ``frame_index = frame - 1`` (0-based, so gt frame ``N`` is image ``N-1.jpg``)
- ``x_min = bb_left - 1``, ``y_min = bb_top - 1``, ``x_max = x_min + w``, ``y_max = y_min + h``
- ``track_id = id`` (an identifier, kept as given)

Parsing checks syntax only. Semantic checks (positive sizes, frame range, duplicates,
bounds) belong to :mod:`passagewatch.validation`, so that problems are reported rather
than silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MOT_COLUMNS = 10
# Trailing columns written for ground-truth-style rows: conf=1, and unused 3D x, y, z.
MOT_TRAILER = "1.0,-1,-1,-1"


class MotFormatError(ValueError):
    """A MOT file row cannot be parsed."""

    def __init__(self, path: Path, line_number: int, reason: str) -> None:
        super().__init__(f"{path}:{line_number}: {reason}")
        self.path = path
        self.line_number = line_number
        self.reason = reason


@dataclass(frozen=True, eq=False)
class BoxAnnotations:
    """Boxes of one clip in the internal convention, one row per (frame, track) box.

    ``boxes`` has shape ``(n, 4)`` with columns ``x_min, y_min, x_max, y_max`` in
    absolute, 0-based continuous pixel coordinates.
    """

    frame_index: NDArray[np.int64]
    track_id: NDArray[np.int64]
    boxes: NDArray[np.float64]

    def __post_init__(self) -> None:
        n = len(self.frame_index)
        if len(self.track_id) != n or self.boxes.shape != (n, 4):
            raise ValueError(
                f"inconsistent shapes: frame_index {self.frame_index.shape}, "
                f"track_id {self.track_id.shape}, boxes {self.boxes.shape}"
            )

    def __len__(self) -> int:
        return len(self.frame_index)

    @classmethod
    def empty(cls) -> BoxAnnotations:
        return cls(
            frame_index=np.empty(0, dtype=np.int64),
            track_id=np.empty(0, dtype=np.int64),
            boxes=np.empty((0, 4), dtype=np.float64),
        )

    def select(self, mask: NDArray[np.bool_]) -> BoxAnnotations:
        return BoxAnnotations(
            frame_index=self.frame_index[mask],
            track_id=self.track_id[mask],
            boxes=self.boxes[mask],
        )

    def in_frame_range(self, start: int, stop: int) -> BoxAnnotations:
        """Rows with ``start <= frame_index < stop``."""
        return self.select((self.frame_index >= start) & (self.frame_index < stop))

    def equals(self, other: BoxAnnotations) -> bool:
        return (
            np.array_equal(self.frame_index, other.frame_index)
            and np.array_equal(self.track_id, other.track_id)
            and np.array_equal(self.boxes, other.boxes)
        )


def mot_to_internal(rows: NDArray[np.float64]) -> BoxAnnotations:
    """Convert ``(n, 6)`` MOT rows ``frame, id, bb_left, bb_top, w, h`` (1-based)."""
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError(f"expected (n, 6) MOT rows, got shape {rows.shape}")
    x_min = rows[:, 2] - 1.0
    y_min = rows[:, 3] - 1.0
    boxes = np.stack([x_min, y_min, x_min + rows[:, 4], y_min + rows[:, 5]], axis=1)
    return BoxAnnotations(
        frame_index=rows[:, 0].astype(np.int64) - 1,
        track_id=rows[:, 1].astype(np.int64),
        boxes=boxes.astype(np.float64),
    )


def internal_to_mot(annotations: BoxAnnotations) -> NDArray[np.float64]:
    """Exact inverse of :func:`mot_to_internal`: ``(n, 6)`` rows, 1-based."""
    b = annotations.boxes
    return np.stack(
        [
            (annotations.frame_index + 1).astype(np.float64),
            annotations.track_id.astype(np.float64),
            b[:, 0] + 1.0,
            b[:, 1] + 1.0,
            b[:, 2] - b[:, 0],
            b[:, 3] - b[:, 1],
        ],
        axis=1,
    )


def _parse_int(token: str) -> int:
    # CFC writes frames and IDs as plain integers; tolerate "12.0" but not "12.5".
    try:
        return int(token)
    except ValueError:
        value = float(token)
        if not value.is_integer():
            raise ValueError(f"not an integer: {token!r}") from None
        return int(value)


def read_mot_rows(path: Path) -> NDArray[np.float64]:
    """Read a MOT file into ``(n, 6)`` rows ``frame, id, bb_left, bb_top, w, h`` (1-based).

    Blank lines are ignored. The trailing ``conf, x, y, z`` columns must be present
    but are not used.
    """
    rows: list[tuple[float, ...]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            fields = text.split(",")
            if len(fields) != MOT_COLUMNS:
                raise MotFormatError(
                    path, line_number, f"expected {MOT_COLUMNS} columns, got {len(fields)}"
                )
            try:
                frame = _parse_int(fields[0])
                track = _parse_int(fields[1])
                box = [float(v) for v in fields[2:6]]
            except ValueError as exc:
                raise MotFormatError(path, line_number, str(exc)) from None
            if not all(math.isfinite(v) for v in box):
                raise MotFormatError(path, line_number, "non-finite box value")
            rows.append((float(frame), float(track), *box))
    if not rows:
        return np.empty((0, 6), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def read_mot(path: Path) -> BoxAnnotations:
    """Read a MOT file and convert it to the internal convention."""
    return mot_to_internal(read_mot_rows(path))


def write_mot(annotations: BoxAnnotations, path: Path) -> None:
    """Write annotations back in MOT format (e.g. for the official CFC evaluator)."""
    lines = [
        f"{int(r[0])},{int(r[1])},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f},{MOT_TRAILER}\n"
        for r in internal_to_mot(annotations)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
