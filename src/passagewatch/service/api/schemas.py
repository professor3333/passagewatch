"""Request and response models of the v1 HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClipOut(_Model):
    clip_id: str
    sha256: str
    media_kind: Literal["frames", "video"]
    num_frames: int
    width: int
    height: int
    framerate: float
    duration_seconds: float
    meters: dict[str, float]
    created_at: str
    expires_at: str | None


class CountingIn(_Model):
    policy: Literal["cfc-compatible-v1"] = "cfc-compatible-v1"
    line_x_normalized: float = Field(default=0.5, gt=0, lt=1)
    upstream_direction: Literal["left", "right"] | None = None


class JobIn(_Model):
    clip_id: str
    counting: CountingIn = CountingIn()
    pipeline_alias: Literal["production"] = "production"


class JobAccepted(_Model):
    job_id: str
    status: str
    pipeline_version: str
    status_url: str


class JobOut(_Model):
    job_id: str
    clip_id: str
    status: str
    progress: float
    attempts: int
    max_attempts: int
    error: str | None
    pipeline_version: str
    counting: dict[str, Any]
    cached_from: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    results_url: str | None


class DirectionalCountsOut(_Model):
    """Image-space counts always; river directions only when orientation is configured."""

    right: int
    left: int
    upstream: int | None
    downstream: int | None
    net_upstream: int | None


class ReviewStateOut(_Model):
    revision: int
    kind: Literal["automatic", "reviewed"]
    state: Literal["pending", "reviewed"]


class ResultsOut(_Model):
    job_id: str
    clip_id: str
    recording_sha256: str
    status: str
    cached: bool
    counting: dict[str, Any]
    pipeline_version: str
    pipeline_config_sha256: str
    automatic: DirectionalCountsOut
    reviewed: DirectionalCountsOut | None
    review: ReviewStateOut
    tracks: int
    tracks_url: str
    created_at: str


class TracksOut(_Model):
    job_id: str
    revision: int
    total: int
    offset: int
    limit: int
    tracks: list[dict[str, Any]]


class ModelInfoOut(_Model):
    pipeline_version: str
    pipeline_config_sha256: str
    detector: dict[str, Any]
    preprocessing_version: str
    tracker: dict[str, Any]
    counting_policy: str
    calibration_version: str | None
    provenance: dict[str, Any]


class ReviewIn(_Model):
    """One correction. ``base_revision`` must be the job's latest revision (else 409)."""

    base_revision: int = Field(ge=0)
    action: Literal["accept", "reject", "set_direction", "mark_unresolved", "add_passage"]
    track_id: int | None = None
    passage_id: str | None = None
    direction: Literal["right", "left"] | None = None
    frame_index: int | None = Field(default=None, ge=0)
    reason: str = Field(default="", max_length=500)


class ReviewOut(_Model):
    job_id: str
    revision: int
    reviewed: DirectionalCountsOut
    unresolved: int
    added_passages: int
    results_url: str


class ObservationsOut(_Model):
    job_id: str
    start: int
    stop: int
    boxes: list[dict[str, Any]]
