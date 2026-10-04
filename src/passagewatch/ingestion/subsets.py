"""Named subsets of CFC imagery, selected by location and whole recording days.

A subset config (``configs/data/*_subset.yaml``) names an archive and the clips to keep.
:class:`CfcFrameSelector` maps archive members ``<prefix><clip>_<n>.jpg`` to
``<location>/<clip>/<n>.jpg``, the frame layout that :class:`CfcLayout` reads.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from passagewatch.ingestion.cfc import FRAME_SUFFIX, LOCATIONS
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.splits import parse_clip_name

# The locations whose clips each CFC v1.1 image archive holds (all members, selected or not).
ARCHIVE_LOCATIONS: dict[str, tuple[str, ...]] = {
    "kenai.tar": ("kenai-train", "kenai-val"),
    "rightbank.tar": ("kenai-rightbank",),
    "channel.tar": ("kenai-channel",),
    "elwha.tar": ("elwha",),
    "nushagak.tar": ("nushagak",),
}
ANY_TOP_DIRECTORY = "*/"  # member_prefix: frames under any single top-level directory


def archive_locations(archive: str) -> tuple[str, ...]:
    key = archive.rpartition("/")[2]
    try:
        return ARCHIVE_LOCATIONS[key]
    except KeyError:
        raise ValueError(f"unknown CFC image archive {archive!r}") from None


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Include(_Frozen):
    location: str
    # None keeps every day of the location.
    days: tuple[dt.date, ...] | None = Field(default=None, min_length=1)

    def model_post_init(self, _context: object) -> None:
        if self.location not in LOCATIONS:
            raise ValueError(f"unknown location {self.location!r}")


class SubsetConfig(_Frozen):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    archive: str
    member_prefix: str
    include: tuple[Include, ...] = Field(min_length=1)
    # "holdout": an internal holdout carved out of kenai-train by whole recording days. Its
    # clips get partition `holdout` in manifests, so training and tuning never select them.
    partition: Literal["holdout"] | None = None

    def model_post_init(self, _context: object) -> None:
        if self.partition == "holdout" and any(
            i.location != "kenai-train" or i.days is None for i in self.include
        ):
            raise ValueError("a holdout subset takes whole days of kenai-train only")

    def describe(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def load_subset_config(path: Path) -> SubsetConfig:
    with path.open("r", encoding="utf-8") as fh:
        return SubsetConfig.model_validate(yaml.safe_load(fh))


def select_clips(
    config: SubsetConfig, metadata: dict[str, dict[str, ClipMetadata]]
) -> dict[str, str]:
    """Selected clip names mapped to their location. ``metadata`` is keyed by location."""
    selected: dict[str, str] = {}
    for include in config.include:
        days = set(include.days) if include.days is not None else None
        clips = metadata[include.location]
        if days is not None:
            present = {parse_clip_name(name).recording_date for name in clips}
            missing = sorted(days - present)
            if missing:
                raise ValueError(f"{include.location} has no clips on {missing}")
        for name in clips:
            if days is None or parse_clip_name(name).recording_date in days:
                selected[name] = include.location
    return selected


class CfcFrameSelector:
    """Chooses archive members for the selected clips, and counts everything it skips."""

    def __init__(
        self, prefix: str, selected: dict[str, str], known_clips: set[str] | frozenset[str]
    ) -> None:
        self.prefix = prefix
        self.selected = selected
        self.known_clips = known_clips
        self.skipped: Counter[str] = Counter()
        self.unknown_examples: list[str] = []
        self.frames: Counter[str] = Counter()

    def _unknown(self, reason: str, name: str) -> None:
        self.skipped[reason] += 1
        if len(self.unknown_examples) < 10:
            self.unknown_examples.append(name)

    def __call__(self, name: str) -> PurePosixPath | None:
        if self.prefix == ANY_TOP_DIRECTORY:
            top, sep, _rest = name.partition("/")
            prefix = f"{top}/" if sep and top else None
        else:
            prefix = self.prefix if name.startswith(self.prefix) else None
        if prefix is None or not name.endswith(FRAME_SUFFIX):
            self._unknown("unexpected_name", name)
            return None
        stem = name.removeprefix(prefix).removesuffix(FRAME_SUFFIX)
        clip, _, frame = stem.rpartition("_")
        if "/" in stem or not clip or not frame.isdigit():
            self._unknown("unexpected_name", name)
            return None
        if clip not in self.known_clips:
            self._unknown("unknown_clip", name)
            return None
        location = self.selected.get(clip)
        if location is None:
            self.skipped["not_selected"] += 1
            return None
        self.frames[clip] += 1
        return PurePosixPath(location, clip, f"{int(frame)}{FRAME_SUFFIX}")
