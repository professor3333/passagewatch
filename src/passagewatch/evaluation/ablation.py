"""One-factor pipeline ablations on cached detections.

A :class:`PipelineVariant` is everything after the detector: score threshold, duplicate
suppression and tracker. :func:`evaluate_variant` runs it on every clip and returns both
counting errors (for nMAE and the paired bootstrap) and passage-level error outcomes
(:mod:`passagewatch.evaluation.errors`), because count errors can cancel within a clip.

A plan (``configs/tracking/tuning/*.yaml``) names a base and a few variants, each
overriding only what it changes. :func:`select` picks the variant with the lowest nMAE;
ties keep the current pipeline, because a change needs a measured benefit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from passagewatch.counting.policy import CFC_COMPATIBLE_V1, CountingPolicy
from passagewatch.detection.classical import FrameDetections
from passagewatch.detection.suppression import SuppressionConfig, suppress_overlaps
from passagewatch.evaluation.errors import ClipErrors, analyze_clip, combine
from passagewatch.evaluation.nmae import ClipCountError, count_clip
from passagewatch.inference.classical import load_classical_config
from passagewatch.inference.neural import above, track_clip
from passagewatch.ingestion.cfc import Clip
from passagewatch.tracking.kalman import TrackerConfig, trajectories_to_annotations


class PipelineVariant(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    score_threshold: float = Field(gt=0, lt=1)
    suppression: SuppressionConfig = SuppressionConfig()
    tracker: TrackerConfig


def plan_variants(plan: Mapping[str, Any], repo_root: Path) -> list[PipelineVariant]:
    """Variants of a plan; each overrides ``score_threshold``, ``suppression`` or tracker
    fields of the base (``tracking_config`` and ``score_threshold``)."""
    base = plan["base"]
    tracker = load_classical_config(repo_root / base["tracking_config"]).tracker
    variants = []
    for name, overrides in plan["variants"].items():
        overrides = dict(overrides or {})
        unknown = set(overrides) - {"score_threshold", "suppression", "tracker"}
        if unknown:
            raise ValueError(f"variant {name}: unknown keys {sorted(unknown)}")
        variants.append(
            PipelineVariant(
                name=name,
                score_threshold=overrides.get("score_threshold", base["score_threshold"]),
                suppression=SuppressionConfig.model_validate(overrides.get("suppression", {})),
                tracker=TrackerConfig.model_validate(
                    tracker.model_dump() | overrides.get("tracker", {})
                ),
            )
        )
    if not variants or variants[0].name != "current":
        raise ValueError("the first variant must be 'current' (the pipeline as released)")
    return variants


def apply_variant(
    detections: list[FrameDetections], variant: PipelineVariant
) -> list[FrameDetections]:
    return [
        suppress_overlaps(d, variant.suppression)
        for d in above(detections, variant.score_threshold)
    ]


@dataclass(frozen=True)
class VariantResult:
    variant: PipelineVariant
    errors: list[ClipCountError]
    passages: dict[str, Any]  # combine() of the passage-level error analysis

    @property
    def absolute_error(self) -> int:
        return sum(e.absolute_error for e in self.errors)

    @property
    def nmae(self) -> float:
        return self.absolute_error / sum(e.reference.total for e in self.errors)

    @property
    def passage_errors(self) -> int:
        ref: dict[str, int] = self.passages["reference_passages"]
        pred: dict[str, int] = self.passages["predicted_passages"]
        return sum(n for k, n in [*ref.items(), *pred.items()] if k != "counted")

    def counts(self) -> tuple[list[list[int]], list[list[int]]]:
        """Per-clip [right, left] reference and predicted counts, for the bootstrap."""
        reference = [[e.reference.right, e.reference.left] for e in self.errors]
        predicted = [[e.predicted.right, e.predicted.left] for e in self.errors]
        return reference, predicted


def evaluate_variant(
    clips: Sequence[Clip],
    cached: Mapping[str, list[FrameDetections]],
    variant: PipelineVariant,
    policy: CountingPolicy = CFC_COMPATIBLE_V1,
) -> VariantResult:
    errors: list[ClipCountError] = []
    analyses: list[ClipErrors] = []
    for clip in clips:
        meta = clip.metadata
        detections = apply_variant(cached[clip.name], variant)
        tracks = trajectories_to_annotations(track_clip(clip, detections, variant.tracker))
        errors.append(
            ClipCountError(
                clip.name,
                count_clip(clip.annotations, meta, policy),
                count_clip(tracks, meta, policy),
                clip.num_window_frames / meta.framerate,
            )
        )
        analyses.append(
            analyze_clip(clip.name, clip.annotations, tracks, meta.width, meta.height, policy)
        )
    return VariantResult(variant, errors, combine(analyses))


def select(results: Sequence[VariantResult]) -> VariantResult:
    """Lowest nMAE, then fewest passage-level errors; ties keep the current pipeline."""
    return min(results, key=lambda r: (r.nmae, r.passage_errors, r.variant.name != "current"))
