"""Error analysis: why counts are wrong, trajectory by trajectory.

Predicted and reference trajectories are matched when they share at least ``min_shared``
frames in which their boxes overlap with IoU >= ``iou``. Then every **reference passage**
(a reference trajectory that the counting policy counts) gets one outcome:

- ``counted`` - a matched predicted trajectory counts the same direction;
- ``wrong_direction`` - a matched predicted trajectory counts the opposite direction;
- ``missed_fish`` - no predicted trajectory matches it at all;
- ``merged_track`` - its only matching predicted trajectory also follows another fish, and
  does not count this passage;
- ``split_track`` - two or more predicted fragments match it, and none counts the passage;
- ``ambiguous_start_end`` - one predicted trajectory matches it, but starts or ends within
  ``line_margin`` of the counting line, so the passage decision depends on where the track
  happens to begin or end;
- ``partial_track`` - one predicted trajectory matches it, but covers too little of its path
  to cross the line.

Every **predicted passage** gets one outcome:

- ``counted`` - matched to a reference passage in the same direction;
- ``duplicate`` - that reference passage is counted by another predicted trajectory that
  shares more frames with it (one fish counted twice);
- ``wrong_direction``;
- ``background`` - matches no reference trajectory (clutter or debris);
- ``non_passing_fish`` - matches a real fish that does not pass.

``corrupted timing/config`` from the project's error categories cannot occur on CFC clips
(fixed frame rates and configurations); it is relevant for uploads only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from passagewatch.counting.policy import CountingPolicy, TrajectoryCount, count_trajectories
from passagewatch.evaluation.detection import pairwise_iou
from passagewatch.ingestion.mot import BoxAnnotations

REFERENCE_OUTCOMES = (
    "counted",
    "wrong_direction",
    "missed_fish",
    "merged_track",
    "split_track",
    "ambiguous_start_end",
    "partial_track",
)
PREDICTED_OUTCOMES = ("counted", "duplicate", "wrong_direction", "background", "non_passing_fish")
DIRECTIONS = ("right", "left", "none")


def shared_frames(
    reference: BoxAnnotations, predicted: BoxAnnotations, iou: float
) -> Counter[tuple[int, int]]:
    """Frames, per (reference track, predicted track), in which their boxes overlap."""
    counts: Counter[tuple[int, int]] = Counter()
    common = np.intersect1d(reference.frame_index, predicted.frame_index)
    for frame in common.tolist():
        r = reference.frame_index == frame
        p = predicted.frame_index == frame
        overlap = pairwise_iou(reference.boxes[r], predicted.boxes[p])
        rows, cols = np.nonzero(overlap >= iou)
        r_ids, p_ids = reference.track_id[r], predicted.track_id[p]
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            counts[(int(r_ids[i]), int(p_ids[j]))] += 1
    return counts


def match_tracks(
    reference: BoxAnnotations,
    predicted: BoxAnnotations,
    *,
    iou: float = 0.3,
    min_shared: int = 3,
) -> dict[int, set[int]]:
    """Reference track ID -> IDs of the predicted tracks that follow it."""
    matches: dict[int, set[int]] = {}
    for (ref_id, pred_id), n in shared_frames(reference, predicted, iou).items():
        if n >= min_shared:
            matches.setdefault(ref_id, set()).add(pred_id)
    return matches


def _direction(count: TrajectoryCount) -> str:
    return "none" if count.direction is None else count.direction.value


@dataclass
class ClipErrors:
    clip_name: str
    reference: Counter[str] = field(default_factory=Counter)
    predicted: Counter[str] = field(default_factory=Counter)
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)
    examples: list[dict[str, object]] = field(default_factory=list)


def analyze_clip(
    clip_name: str,
    reference: BoxAnnotations,
    predicted: BoxAnnotations,
    width: int,
    height: int,
    policy: CountingPolicy,
    *,
    iou: float = 0.3,
    min_shared: int = 3,
    line_margin: float = 0.05,
) -> ClipErrors:
    ref_counts = {c.track_id: c for c in count_trajectories(reference, width, height, policy)}
    pred_counts = {c.track_id: c for c in count_trajectories(predicted, width, height, policy)}
    matches = match_tracks(reference, predicted, iou=iou, min_shared=min_shared)
    followed: dict[int, set[int]] = {}
    for ref_id, pred_ids in matches.items():
        for pred_id in pred_ids:
            followed.setdefault(pred_id, set()).add(ref_id)
    result = ClipErrors(clip_name)
    line = policy.line_x_normalized

    shared = shared_frames(reference, predicted, iou)

    # One-to-one attribution: each predicted passage goes to the same-direction reference
    # passage it shares the most frames with; per reference passage, the predicted passage
    # with the most shared frames counts it and any others are duplicates.
    attributed: dict[int, list[int]] = {}
    for pred_id, pred in pred_counts.items():
        if pred.direction is None:
            continue
        candidates = [
            r
            for r in followed.get(pred_id, ())
            if r in ref_counts and ref_counts[r].direction == pred.direction
        ]
        if candidates:
            best = max(candidates, key=lambda r: (shared[(r, pred_id)], -r))
            attributed.setdefault(best, []).append(pred_id)
    for ref_id, preds in attributed.items():
        preds.sort(key=lambda p: (-shared[(ref_id, p)], p))

    for ref_id, ref in ref_counts.items():
        if ref.direction is None:
            continue
        matched = [pred_counts[p] for p in sorted(matches.get(ref_id, ())) if p in pred_counts]
        opposite = [p for p in matched if p.direction is not None and p.direction != ref.direction]
        if ref_id in attributed:
            outcome = "counted"
        elif opposite:
            outcome = "wrong_direction"
        elif not matched:
            outcome = "missed_fish"
        elif any(len(followed.get(p.track_id, ())) >= 2 for p in matched):
            outcome = "merged_track"
        elif len(matched) >= 2:
            outcome = "split_track"
        elif min(abs(matched[0].start_u - line), abs(matched[0].end_u - line)) < line_margin:
            outcome = "ambiguous_start_end"
        else:
            outcome = "partial_track"
        result.reference[outcome] += 1
        if outcome != "counted":
            result.examples.append(
                {
                    "clip_name": clip_name,
                    "kind": "reference",
                    "outcome": outcome,
                    "track_id": ref_id,
                    "frames": [ref.start_frame, ref.end_frame],
                    "direction": _direction(ref),
                }
            )

    counted_by = {preds[0]: ref for ref, preds in attributed.items()}
    duplicates = {p for preds in attributed.values() for p in preds[1:]}
    for pred_id, pred in pred_counts.items():
        if pred.direction is None:
            continue
        refs = [ref_counts[r] for r in followed.get(pred_id, ()) if r in ref_counts]
        if pred_id in counted_by:
            outcome = "counted"
        elif pred_id in duplicates:
            outcome = "duplicate"
        elif any(r.direction is not None for r in refs):
            outcome = "wrong_direction"
        elif not refs:
            outcome = "background"
        else:
            outcome = "non_passing_fish"
        result.predicted[outcome] += 1
        if outcome in ("duplicate", "background", "non_passing_fish"):
            result.examples.append(
                {
                    "clip_name": clip_name,
                    "kind": "predicted",
                    "outcome": outcome,
                    "track_id": pred_id,
                    "frames": [pred.start_frame, pred.end_frame],
                    "direction": _direction(pred),
                }
            )

    # Direction confusion over reference trajectories (each with its best-matching
    # predicted trajectory), plus predicted passages that follow no reference trajectory.
    for ref_id, ref in ref_counts.items():
        candidates = [p for p in matches.get(ref_id, ()) if p in pred_counts]
        if candidates:
            best = max(candidates, key=lambda p: (shared[(ref_id, p)], -p))
            predicted_direction = _direction(pred_counts[best])
        else:
            predicted_direction = "none"
        result.confusion[(_direction(ref), predicted_direction)] += 1
    for pred_id, pred in pred_counts.items():
        if pred.direction is not None and not followed.get(pred_id):
            result.confusion[("none", _direction(pred))] += 1
    return result


def combine(results: list[ClipErrors]) -> dict[str, object]:
    reference: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    for r in results:
        reference.update(r.reference)
        predicted.update(r.predicted)
        confusion.update(r.confusion)
    return {
        "reference_passages": {k: reference[k] for k in REFERENCE_OUTCOMES},
        "predicted_passages": {k: predicted[k] for k in PREDICTED_OUTCOMES},
        "direction_confusion": {
            f"reference_{a}": {f"predicted_{b}": confusion[(a, b)] for b in DIRECTIONS}
            for a in DIRECTIONS
        },
    }


def recall_by_size(
    reference: BoxAnnotations,
    detected: list[np.ndarray],
    frame_start: int,
    area_m2_per_px: float,
    bins: tuple[float, ...],
    iou: float = 0.3,
) -> tuple[Counter[int], Counter[int]]:
    """Reference boxes per size bin, and how many of them a detection overlaps (IoU >= iou).

    ``detected[t]`` are the detection boxes of frame ``frame_start + t``. ``bins`` are the
    upper edges of all bins but the last, in m².
    """
    total: Counter[int] = Counter()
    found: Counter[int] = Counter()
    areas = (
        (reference.boxes[:, 2] - reference.boxes[:, 0])
        * (reference.boxes[:, 3] - reference.boxes[:, 1])
        * area_m2_per_px
    )
    bin_of = np.searchsorted(np.asarray(bins), areas, side="right")
    for t, boxes in enumerate(detected):
        rows = np.flatnonzero(reference.frame_index == frame_start + t)
        if len(rows) == 0:
            continue
        hit = (
            pairwise_iou(reference.boxes[rows], boxes).max(axis=1) >= iou
            if len(boxes)
            else np.zeros(len(rows), dtype=bool)
        )
        for row, h in zip(rows.tolist(), hit.tolist(), strict=True):
            total[int(bin_of[row])] += 1
            found[int(bin_of[row])] += int(h)
    return total, found
