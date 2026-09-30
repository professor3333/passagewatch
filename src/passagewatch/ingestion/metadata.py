"""Per-clip metadata from the CFC ``metadata/<location>.json`` files.

Unknown fields are rejected, so a change in the publisher's format (for example, an
``upstream_direction`` field appearing) is noticed instead of silently ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ClipMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    clip_name: str = Field(min_length=1)
    num_frames: int = Field(gt=0)
    framerate: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    x_meter_start: float
    x_meter_stop: float
    y_meter_start: float
    y_meter_stop: float


@dataclass(frozen=True)
class MetadataFile:
    """The valid clips of one metadata file, and a description of each invalid entry."""

    clips: dict[str, ClipMetadata]
    errors: list[str] = field(default_factory=list)


def load_metadata(path: Path) -> MetadataFile:
    """Load one location's metadata, validating each entry on its own.

    An invalid or duplicated entry is reported in ``errors`` and left out of ``clips``;
    it does not prevent the other clips from loading.
    """
    with path.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a JSON list of clips, got {type(entries).__name__}")

    clips: dict[str, ClipMetadata] = {}
    errors: list[str] = []
    duplicates: set[str] = set()
    for index, entry in enumerate(entries):
        try:
            clip = ClipMetadata.model_validate(entry)
        except ValidationError as exc:
            name = entry.get("clip_name") if isinstance(entry, dict) else None
            reasons = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            errors.append(f"entry {index} ({name!r}): {reasons}")
            continue
        if clip.clip_name in clips or clip.clip_name in duplicates:
            duplicates.add(clip.clip_name)
            errors.append(f"entry {index}: duplicate clip_name {clip.clip_name!r}")
            continue
        clips[clip.clip_name] = clip
    # An ambiguous clip is not trusted at all.
    for name in duplicates:
        clips.pop(name, None)
    return MetadataFile(clips=clips, errors=errors)
