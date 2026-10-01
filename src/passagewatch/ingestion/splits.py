"""Partition assignment for CFC clips, and the recording groups encoded in clip names.

Rules (``docs/dataset_card.md``, "Splits"):

- The publisher's separation is kept: a clip's partition is its location's official split.
  Whole clips are assigned; frames are never split.
- Clips from the official test locations are never used for training, tuning, threshold
  selection, or error mining (``tuning_allowed = False``).
- Every CFC clip name is ``<recording>_<start>_<stop>``: frames ``[start, stop)`` of one
  source recording whose name ends in ``YYYY-MM-DD_HHMMSS``. Internal holdouts carved out
  of a partition must group clips by ``recording_id`` or ``recording_date``, never by clip.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from enum import StrEnum


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


OFFICIAL_SPLITS: dict[str, Split] = {
    "kenai-train": Split.TRAIN,
    "kenai-val": Split.VAL,
    "kenai-rightbank": Split.TEST,
    "kenai-channel": Split.TEST,
    "elwha": Split.TEST,
    "nushagak": Split.TEST,
}

_CLIP_NAME = re.compile(
    r"^(?P<recording>.*?(?P<date>\d{4}-\d{2}-\d{2})_\d{6})_(?P<start>\d+)_(?P<stop>\d+)$"
)


@dataclass(frozen=True)
class ClipName:
    recording_id: str
    recording_date: dt.date
    start: int
    stop: int


def parse_clip_name(name: str) -> ClipName:
    match = _CLIP_NAME.match(name)
    if match is None:
        raise ValueError(f"clip name does not match <recording>_<start>_<stop>: {name!r}")
    start, stop = int(match["start"]), int(match["stop"])
    if stop <= start:
        raise ValueError(f"clip name has an empty frame range: {name!r}")
    return ClipName(
        recording_id=match["recording"],
        recording_date=dt.date.fromisoformat(match["date"]),
        start=start,
        stop=stop,
    )


def official_split(location: str) -> Split:
    try:
        return OFFICIAL_SPLITS[location]
    except KeyError:
        raise ValueError(f"unknown CFC location {location!r}") from None


def tuning_allowed(split: Split) -> bool:
    return split is not Split.TEST
