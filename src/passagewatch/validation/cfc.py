"""Validation of extracted CFC clips.

Every problem becomes an :class:`Issue` with a fixed code. A clip with any ERROR issue is
**quarantined**: it is kept on disk and listed with its reasons, but later stages must not
use it. WARNING issues are recorded and the clip stays usable. Nothing is skipped or
repaired silently. The issue codes are documented in ``docs/dataset_card.md``.

Box bounds are checked against the metadata frame size, which is what the official
evaluator normalizes by. Boxes are kept unclipped.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from passagewatch.ingestion.cfc import (
    GT_FILE,
    LOCATIONS,
    TINY_GT_FILE,
    CfcLayout,
    frame_indices,
    frame_path,
)
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.mot import (
    BoxAnnotations,
    MotFormatError,
    mot_to_internal,
    read_mot_rows,
)
from passagewatch.ingestion.splits import parse_clip_name

MAX_EXAMPLES = 5
REPORT_VERSION = 1


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Code(StrEnum):
    # Errors: the clip is quarantined.
    MISSING_METADATA_FILE = "missing_metadata_file"
    INVALID_METADATA = "invalid_metadata"
    MISSING_METADATA = "missing_metadata"
    MISSING_ANNOTATIONS = "missing_annotations"
    MISSING_CLIP_DIRECTORY = "missing_clip_directory"
    MOT_FORMAT = "mot_format"
    FRAME_OUT_OF_RANGE = "frame_out_of_range"
    DUPLICATE_TRACK_FRAME = "duplicate_track_frame"
    NON_POSITIVE_BOX_SIZE = "non_positive_box_size"
    BOX_OUTSIDE_IMAGE = "box_outside_image"
    MISSING_FRAMES = "missing_frames"
    FRAME_WINDOW_OUT_OF_RANGE = "frame_window_out_of_range"
    UNREADABLE_FRAME = "unreadable_frame"
    IMAGE_SIZE_MISMATCH = "image_size_mismatch"
    TINY_ROWS_NOT_IN_GT = "tiny_rows_not_in_gt"
    INVALID_CLIP_NAME = "invalid_clip_name"
    # Warnings: recorded; the clip stays usable.
    BOX_PARTIALLY_OUTSIDE_IMAGE = "box_partially_outside_image"
    IMAGE_SIZE_DIFFERS_SLIGHTLY = "image_size_differs_slightly"
    UNEXPECTED_FILE = "unexpected_file"
    MISSING_TINY_ANNOTATIONS = "missing_tiny_annotations"
    NO_BOXES_IN_WINDOW = "no_boxes_in_window"


WARNING_CODES = frozenset(
    {
        Code.BOX_PARTIALLY_OUTSIDE_IMAGE,
        Code.IMAGE_SIZE_DIFFERS_SLIGHTLY,
        Code.UNEXPECTED_FILE,
        Code.MISSING_TINY_ANNOTATIONS,
        Code.NO_BOXES_IN_WINDOW,
    }
)


def severity_of(code: Code) -> Severity:
    return Severity.WARNING if code in WARNING_CODES else Severity.ERROR


@dataclass(frozen=True)
class Issue:
    code: Code
    severity: Severity
    message: str
    count: int = 1
    examples: tuple[str, ...] = ()


def _issue(code: Code, message: str, examples: Iterable[str] = (), count: int = 1) -> Issue:
    return Issue(
        code=code,
        severity=severity_of(code),
        message=message,
        count=count,
        examples=tuple(examples)[:MAX_EXAMPLES],
    )


@dataclass(frozen=True)
class ClipStats:
    frame_start: int
    frame_stop: int
    boxes: int
    tracks: int
    annotated_frames: int
    tracks_with_gaps: int
    single_observation_tracks: int
    # Tracks with boxes both inside and outside the frame window (tiny subset only).
    tracks_cut_by_window: int
    frames_checked: int
    image_sizes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ClipReport:
    location: str
    clip_name: str
    issues: tuple[Issue, ...]
    stats: ClipStats | None = None

    @property
    def status(self) -> str:
        severities = {i.severity for i in self.issues}
        if Severity.ERROR in severities:
            return "quarantined"
        return "warning" if severities else "ok"

    @property
    def quarantine_reasons(self) -> list[str]:
        return [i.code.value for i in self.issues if i.severity is Severity.ERROR]


@dataclass
class ValidationReport:
    subset: str
    images_checked: bool
    image_size_tolerance_px: int
    clips: list[ClipReport] = field(default_factory=list)
    # Problems that concern a whole location rather than one clip.
    location_issues: dict[str, list[Issue]] = field(default_factory=dict)

    @property
    def quarantined(self) -> list[ClipReport]:
        return [c for c in self.clips if c.status == "quarantined"]

    def summary(self) -> dict[str, Any]:
        per_location: dict[str, Counter[str]] = {}
        for clip in self.clips:
            per_location.setdefault(clip.location, Counter())[clip.status] += 1
        codes: Counter[str] = Counter()
        affected: Counter[str] = Counter()
        for clip in self.clips:
            for issue in clip.issues:
                codes[issue.code.value] += issue.count
                affected[issue.code.value] += 1
        return {
            "clips": len(self.clips),
            "quarantined": len(self.quarantined),
            "by_location": {
                loc: {s: per_location[loc][s] for s in ("ok", "warning", "quarantined")}
                for loc in per_location
            },
            "issue_totals": {c: {"count": codes[c], "clips": affected[c]} for c in sorted(codes)},
        }

    def to_dict(self) -> dict[str, Any]:
        def clip_dict(c: ClipReport) -> dict[str, Any]:
            return {
                "location": c.location,
                "clip_name": c.clip_name,
                "status": c.status,
                "quarantine_reasons": c.quarantine_reasons,
                "issues": [asdict(i) for i in c.issues],
                "stats": None if c.stats is None else asdict(c.stats),
            }

        return {
            "report_version": REPORT_VERSION,
            "dataset": "cfc",
            "subset": self.subset,
            "images_checked": self.images_checked,
            "image_size_tolerance_px": self.image_size_tolerance_px,
            "summary": self.summary(),
            "location_issues": {
                loc: [asdict(i) for i in issues] for loc, issues in self.location_issues.items()
            },
            "clips": [clip_dict(c) for c in self.clips],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # One compact line per clip keeps the committed report small and its diffs readable.
        data = self.to_dict()
        clips = data.pop("clips")
        head = json.dumps(data, indent=1).removesuffix("\n}")
        lines = ",\n".join("  " + json.dumps(c, separators=(",", ":")) for c in clips)
        tmp.write_text(f'{head},\n "clips": [\n{lines}\n ]\n}}\n', encoding="utf-8")
        tmp.replace(path)


def _describe_rows(ann: BoxAnnotations, mask: np.ndarray[Any, np.dtype[np.bool_]]) -> list[str]:
    idx = np.flatnonzero(mask)[:MAX_EXAMPLES]
    return [
        f"frame_index {ann.frame_index[i]} track {ann.track_id[i]} "
        f"box {tuple(round(float(v), 3) for v in ann.boxes[i])}"
        for i in idx
    ]


def _ranges(values: list[int]) -> list[str]:
    """Compress sorted integers into ``a-b`` range strings."""
    runs: list[list[int]] = []
    for v in values:
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return [str(a) if a == b else f"{a}-{b}" for a, b in runs]


def check_boxes(ann: BoxAnnotations, width: int, height: int) -> list[Issue]:
    """Duplicate (frame, track) pairs, non-positive sizes, and boxes outside the image."""
    issues: list[Issue] = []
    if len(ann) == 0:
        return issues

    pairs = np.stack([ann.frame_index, ann.track_id], axis=1)
    _, first, counts = np.unique(pairs, axis=0, return_index=True, return_counts=True)
    duplicated = counts > 1
    if duplicated.any():
        examples = [
            f"frame_index {pairs[i, 0]} track {pairs[i, 1]} x{c}"
            for i, c in zip(first[duplicated], counts[duplicated], strict=True)
        ]
        issues.append(
            _issue(
                Code.DUPLICATE_TRACK_FRAME,
                "a track has more than one box in the same frame",
                examples,
                count=int((counts[duplicated] - 1).sum()),
            )
        )

    b = ann.boxes
    non_positive = (b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])
    if non_positive.any():
        issues.append(
            _issue(
                Code.NON_POSITIVE_BOX_SIZE,
                "box width or height is not positive",
                _describe_rows(ann, non_positive),
                count=int(non_positive.sum()),
            )
        )

    valid = ~non_positive
    outside = valid & ((b[:, 2] <= 0) | (b[:, 3] <= 0) | (b[:, 0] >= width) | (b[:, 1] >= height))
    if outside.any():
        issues.append(
            _issue(
                Code.BOX_OUTSIDE_IMAGE,
                f"box does not overlap the {width}x{height} image",
                _describe_rows(ann, outside),
                count=int(outside.sum()),
            )
        )
    partial = (
        valid & ~outside & ((b[:, 0] < 0) | (b[:, 1] < 0) | (b[:, 2] > width) | (b[:, 3] > height))
    )
    if partial.any():
        issues.append(
            _issue(
                Code.BOX_PARTIALLY_OUTSIDE_IMAGE,
                f"box extends beyond the {width}x{height} image (kept unclipped)",
                _describe_rows(ann, partial),
                count=int(partial.sum()),
            )
        )
    return issues


def _track_stats(ann: BoxAnnotations) -> tuple[int, int, int]:
    """Number of tracks, tracks with missing frames between their ends, single-box tracks."""
    tracks = gaps = singles = 0
    for track in np.unique(ann.track_id):
        frames = np.unique(ann.frame_index[ann.track_id == track])
        tracks += 1
        singles += int(len(frames) == 1)
        gaps += int(frames[-1] - frames[0] + 1 != len(frames))
    return tracks, gaps, singles


def _check_frames(
    frame_dir: Path | None, meta: ClipMetadata, subset: str
) -> tuple[list[Issue], list[int]]:
    """Check the frame files; return issues and the sorted frame indices that exist."""
    if frame_dir is None:
        return [], []
    if not frame_dir.is_dir():
        return [_issue(Code.MISSING_CLIP_DIRECTORY, f"no frame directory {frame_dir}")], []

    issues: list[Issue] = []
    indices, others = frame_indices(frame_dir)
    if others:
        issues.append(
            _issue(
                Code.UNEXPECTED_FILE,
                "non-frame files in the frame directory (ignored)",
                others,
                len(others),
            )
        )
    if not indices:
        issues.append(_issue(Code.MISSING_FRAMES, "the frame directory has no frames"))
        return issues, indices

    # The tiny subset keeps a window of consecutive frames; a full clip keeps all of them.
    tiny = subset == "tiny"
    expected = range(indices[0], indices[-1] + 1) if tiny else range(meta.num_frames)
    present = set(indices)
    missing = [i for i in expected if i not in present]
    if missing:
        issues.append(
            _issue(
                Code.MISSING_FRAMES,
                f"frames missing inside the window [{expected.start}, {expected.stop})",
                _ranges(missing),
                len(missing),
            )
        )
    beyond = [i for i in indices if i >= meta.num_frames]
    if beyond:
        issues.append(
            _issue(
                Code.FRAME_WINDOW_OUT_OF_RANGE,
                f"frame files beyond num_frames={meta.num_frames}",
                _ranges(beyond),
                len(beyond),
            )
        )
    return issues, indices


def _check_images(
    frame_dir: Path, indices: list[int], meta: ClipMetadata, tolerance: int
) -> tuple[list[Issue], set[tuple[int, int]]]:
    """Decode every frame; compare its size with the metadata."""
    unreadable: list[int] = []
    sizes: Counter[tuple[int, int]] = Counter()
    for i in indices:
        image = cv2.imread(str(frame_path(frame_dir, i)), cv2.IMREAD_UNCHANGED)
        if image is None:
            unreadable.append(i)
            continue
        sizes[(int(image.shape[1]), int(image.shape[0]))] += 1

    issues: list[Issue] = []
    if unreadable:
        issues.append(
            _issue(
                Code.UNREADABLE_FRAME,
                "frame could not be decoded",
                _ranges(unreadable),
                len(unreadable),
            )
        )
    expected = (meta.width, meta.height)
    for (w, h), n in sorted(sizes.items()):
        if (w, h) == expected:
            continue
        worst = max(abs(w - meta.width), abs(h - meta.height))
        code = Code.IMAGE_SIZE_DIFFERS_SLIGHTLY if worst <= tolerance else Code.IMAGE_SIZE_MISMATCH
        issues.append(
            _issue(
                code,
                f"image size {w}x{h} differs from metadata {meta.width}x{meta.height}",
                count=n,
            )
        )
    return issues, set(sizes)


def _check_tiny_rows(tiny_path: Path | None, gt_rows: np.ndarray[Any, Any]) -> list[Issue]:
    if tiny_path is None:
        return []
    if not tiny_path.is_file():
        return [_issue(Code.MISSING_TINY_ANNOTATIONS, f"no {TINY_GT_FILE} (not used)")]
    try:
        tiny_rows = read_mot_rows(tiny_path)
    except MotFormatError as exc:
        return [_issue(Code.MOT_FORMAT, str(exc))]
    known = {tuple(r) for r in gt_rows.tolist()}
    extra = [r for r in tiny_rows.tolist() if tuple(r) not in known]
    if not extra:
        return []
    return [
        _issue(
            Code.TINY_ROWS_NOT_IN_GT,
            f"{TINY_GT_FILE} has rows that are not in {GT_FILE}",
            [str(r) for r in extra],
            len(extra),
        )
    ]


def validate_clip(
    layout: CfcLayout,
    location: str,
    meta: ClipMetadata,
    *,
    check_images: bool = True,
    image_size_tolerance_px: int = 1,
) -> ClipReport:
    clip = meta.clip_name
    issues: list[Issue] = []

    def report(stats: ClipStats | None = None) -> ClipReport:
        return ClipReport(location, clip, tuple(issues), stats)

    try:
        name = parse_clip_name(clip)
    except ValueError as exc:
        issues.append(_issue(Code.INVALID_CLIP_NAME, str(exc)))
    else:
        if name.stop - name.start != meta.num_frames:
            issues.append(
                _issue(
                    Code.INVALID_CLIP_NAME,
                    f"clip name covers {name.stop - name.start} frames "
                    f"but num_frames={meta.num_frames}",
                )
            )

    gt_path = layout.gt_path(location, clip)
    if not gt_path.is_file():
        issues.append(_issue(Code.MISSING_ANNOTATIONS, f"no {GT_FILE} for this clip"))
        return report()
    allowed = {GT_FILE}
    stray = sorted(p.name for p in gt_path.parent.iterdir() if p.name not in allowed)
    if stray:
        issues.append(
            _issue(
                Code.UNEXPECTED_FILE,
                "non-annotation files in the annotation directory (ignored)",
                stray,
                len(stray),
            )
        )

    try:
        gt_rows = read_mot_rows(gt_path)
    except MotFormatError as exc:
        issues.append(_issue(Code.MOT_FORMAT, str(exc)))
        return report()
    all_ann = mot_to_internal(gt_rows)

    # Every row must reference a real frame, whichever window is used later.
    out_of_range = (all_ann.frame_index < 0) | (all_ann.frame_index >= meta.num_frames)
    if out_of_range.any():
        issues.append(
            _issue(
                Code.FRAME_OUT_OF_RANGE,
                f"annotation frame outside [1, num_frames={meta.num_frames}] (1-based)",
                _describe_rows(all_ann, out_of_range),
                int(out_of_range.sum()),
            )
        )

    frame_dir = layout.frame_dir(location, clip)
    frame_issues, indices = _check_frames(frame_dir, meta, layout.subset)
    issues.extend(frame_issues)
    if layout.subset == "tiny":
        if not indices:
            return report()
        start, stop = indices[0], indices[-1] + 1
    else:
        start, stop = 0, meta.num_frames
    in_window = (all_ann.frame_index >= start) & (all_ann.frame_index < stop)
    ann = all_ann.select(in_window)

    issues.extend(check_boxes(ann, meta.width, meta.height))
    if layout.subset == "tiny":
        issues.extend(_check_tiny_rows(layout.tiny_gt_path(location, clip), gt_rows))
        if len(ann) == 0:
            issues.append(_issue(Code.NO_BOXES_IN_WINDOW, "no boxes inside the frame window"))

    sizes: set[tuple[int, int]] = set()
    if check_images and frame_dir is not None and indices:
        image_issues, sizes = _check_images(frame_dir, indices, meta, image_size_tolerance_px)
        issues.extend(image_issues)

    tracks, gaps, singles = _track_stats(ann)
    window_tracks = set(np.unique(ann.track_id).tolist())
    outside_tracks = set(np.unique(all_ann.track_id[~in_window]).tolist())
    stats = ClipStats(
        frame_start=start,
        frame_stop=stop,
        boxes=len(ann),
        tracks=tracks,
        annotated_frames=len(np.unique(ann.frame_index)),
        tracks_with_gaps=gaps,
        single_observation_tracks=singles,
        tracks_cut_by_window=len(window_tracks & outside_tracks),
        frames_checked=len(indices) if check_images and frame_dir is not None else 0,
        image_sizes=tuple(sorted(sizes)),
    )
    return report(stats)


def validate_dataset(
    layout: CfcLayout,
    *,
    locations: Iterable[str] = LOCATIONS,
    check_images: bool = True,
    image_size_tolerance_px: int = 1,
) -> ValidationReport:
    report = ValidationReport(
        subset=layout.subset,
        images_checked=check_images and layout.frames_dir is not None,
        image_size_tolerance_px=image_size_tolerance_px,
    )
    for location in locations:
        location_issues: list[Issue] = []
        metadata_path = layout.metadata_dir / f"{location}.json"
        if not metadata_path.is_file():
            report.location_issues[location] = [
                _issue(Code.MISSING_METADATA_FILE, f"no metadata file {metadata_path.name}")
            ]
            continue
        metadata = layout.metadata(location)
        if metadata.errors:
            location_issues.append(
                _issue(
                    Code.INVALID_METADATA,
                    "metadata entries rejected; their clips are quarantined",
                    metadata.errors,
                    len(metadata.errors),
                )
            )
        if location_issues:
            report.location_issues[location] = location_issues

        present = layout.clip_names(location)
        for clip in sorted(present | set(metadata.clips)):
            meta = metadata.clips.get(clip)
            if meta is None:
                issue = _issue(Code.MISSING_METADATA, "clip directory has no valid metadata")
                report.clips.append(ClipReport(location, clip, (issue,)))
            elif clip not in present:
                code = Code.MISSING_CLIP_DIRECTORY
                if layout.subset == "full":
                    code = Code.MISSING_ANNOTATIONS
                issue = _issue(code, "metadata entry has no clip directory")
                report.clips.append(ClipReport(location, clip, (issue,)))
            else:
                report.clips.append(
                    validate_clip(
                        layout,
                        location,
                        meta,
                        check_images=check_images,
                        image_size_tolerance_px=image_size_tolerance_px,
                    )
                )
    return report
