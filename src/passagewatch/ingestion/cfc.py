"""Directory layout of extracted CFC data, and loading one clip in the internal convention.

Two layouts are supported:

- **full**: ``annotations/<location>/<clip>/gt.txt`` and ``metadata/<location>.json``,
  optionally with frames at ``<frames>/<location>/<clip>/<n>.jpg`` for ``n`` in
  ``[0, num_frames)``.
- **tiny**: the CFC tiny subset, ``raw/<location>/<clip>/<n>.jpg`` for a window of
  consecutive frames. Its ``gt_tiny.txt`` is **not** aligned with that window (see
  ``docs/dataset_card.md``), so tiny clips take their boxes from the full ``gt.txt``,
  restricted to the image window. ``gt_tiny.txt`` is only checked for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from passagewatch.ingestion.metadata import ClipMetadata, MetadataFile, load_metadata
from passagewatch.ingestion.mot import BoxAnnotations, read_mot

LOCATIONS = (
    "kenai-train",
    "kenai-val",
    "kenai-rightbank",
    "kenai-channel",
    "elwha",
    "nushagak",
)
GT_FILE = "gt.txt"
TINY_GT_FILE = "gt_tiny.txt"
FRAME_SUFFIX = ".jpg"

Subset = Literal["full", "tiny"]


@dataclass(frozen=True)
class CfcLayout:
    subset: Subset
    annotations_dir: Path
    metadata_dir: Path
    frames_dir: Path | None = None
    tiny_annotations_dir: Path | None = None
    # Full layout only: when set, only these clips are expected to have frames (a subset
    # such as configs/data/kenai_subset.yaml); the others are checked as annotations only.
    frame_clips: frozenset[str] | None = None

    @classmethod
    def full(
        cls,
        extract_root: Path,
        frames_dir: Path | None = None,
        frame_clips: frozenset[str] | None = None,
    ) -> CfcLayout:
        """``extract_root`` is the directory holding the extracted publisher archives."""
        return cls(
            subset="full",
            annotations_dir=extract_root / "fish_counting_annotations/annotations",
            metadata_dir=extract_root / "fish_counting_metadata/metadata",
            frames_dir=frames_dir,
            frame_clips=frame_clips,
        )

    @classmethod
    def tiny(cls, extract_root: Path) -> CfcLayout:
        tiny_root = extract_root / "tiny_dataset/tiny_dataset"
        return cls(
            subset="tiny",
            annotations_dir=extract_root / "fish_counting_annotations/annotations",
            metadata_dir=tiny_root / "metadata-tiny",
            frames_dir=tiny_root / "raw",
            tiny_annotations_dir=tiny_root / "annotations-tiny",
        )

    def metadata(self, location: str) -> MetadataFile:
        return load_metadata(self.metadata_dir / f"{location}.json")

    def clip_names(self, location: str) -> set[str]:
        """Clip directories present for ``location`` in this subset."""
        root = self.frames_dir if self.subset == "tiny" else self.annotations_dir
        assert root is not None
        location_dir = root / location
        if not location_dir.is_dir():
            return set()
        return {p.name for p in location_dir.iterdir() if p.is_dir()}

    def gt_path(self, location: str, clip: str) -> Path:
        return self.annotations_dir / location / clip / GT_FILE

    def tiny_gt_path(self, location: str, clip: str) -> Path | None:
        if self.tiny_annotations_dir is None:
            return None
        return self.tiny_annotations_dir / location / clip / TINY_GT_FILE

    def frame_dir(self, location: str, clip: str) -> Path | None:
        if self.frames_dir is None:
            return None
        if self.frame_clips is not None and clip not in self.frame_clips:
            return None
        return self.frames_dir / location / clip


def frame_indices(frame_dir: Path) -> tuple[list[int], list[str]]:
    """Sorted 0-based frame indices of ``<n>.jpg`` files, and the names of any other files."""
    indices: list[int] = []
    others: list[str] = []
    for path in frame_dir.iterdir():
        stem = path.name.removesuffix(FRAME_SUFFIX)
        if path.is_file() and path.name.endswith(FRAME_SUFFIX) and stem.isdigit():
            indices.append(int(stem))
        else:
            others.append(path.name)
    indices.sort()
    others.sort()
    return indices, others


def frame_path(frame_dir: Path, index: int) -> Path:
    return frame_dir / f"{index}{FRAME_SUFFIX}"


@dataclass(frozen=True)
class Clip:
    """One clip ready for use: metadata, the frame window, and boxes inside that window."""

    location: str
    metadata: ClipMetadata
    frame_start: int
    frame_stop: int
    annotations: BoxAnnotations
    frame_dir: Path | None

    @property
    def name(self) -> str:
        return self.metadata.clip_name

    @property
    def num_window_frames(self) -> int:
        return self.frame_stop - self.frame_start


def load_clip(layout: CfcLayout, location: str, metadata: ClipMetadata) -> Clip:
    """Load one clip's boxes in the internal convention, restricted to its frame window.

    This does not validate the clip; run the validator first and skip quarantined clips.
    For the tiny subset the window is the contiguous range of images present; for the full
    layout it is the whole clip.
    """
    annotations = read_mot(layout.gt_path(location, metadata.clip_name))
    frame_dir = layout.frame_dir(location, metadata.clip_name)
    start, stop = 0, metadata.num_frames
    if layout.subset == "tiny":
        assert frame_dir is not None
        indices, _ = frame_indices(frame_dir)
        if not indices:
            raise ValueError(f"no frames in {frame_dir}")
        start, stop = indices[0], indices[-1] + 1
    return Clip(
        location=location,
        metadata=metadata,
        frame_start=start,
        frame_stop=stop,
        annotations=annotations.in_frame_range(start, stop),
        frame_dir=frame_dir,
    )
