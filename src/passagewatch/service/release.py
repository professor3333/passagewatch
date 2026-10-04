"""Release manifests: what a release is, and how to check that a deployment is that release.

A release manifest (``releases/manifests/<release>.json``, committed) is immutable. It
records everything that determines the release's results and how it was evaluated:

- the code commit (with no uncommitted changes) and the SHA-256 of ``uv.lock``;
- the dataset manifest (version and content hash) and the training run (its config hash and
  training commit);
- the bundle's identity: its pipeline config hash, checkpoint and ONNX hashes, preprocessing
  version, tracker configuration, counting-policy and calibration versions;
- the evaluation summary it was released on, with the hashes of the report files;
- the image digest, once a release workflow has built and pushed the image (``null`` until
  then).

:func:`check_model_info` compares a running service's ``/v1/model-info`` with a manifest
field by field; a deployment is the evaluated release only if nothing differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from passagewatch.service.bundle import ReleaseBundle
from passagewatch.service.jobs import canonical_json

MANIFEST_VERSION = 1


class ManifestExistsError(ValueError):
    """A release manifest already exists with different content."""


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_version: int = MANIFEST_VERSION
    release: str
    created_at: str
    code_commit: str
    uv_lock_sha256: str
    dataset_manifest: dict[str, Any]
    training: dict[str, Any]
    bundle: dict[str, Any]
    evaluation: dict[str, Any]
    image_digest: str | None = None


def bundle_identity(bundle: ReleaseBundle) -> dict[str, Any]:
    """The bundle fields a deployment must match (all part of the pipeline config hash)."""
    return {
        "pipeline_config_sha256": bundle.config_sha256(),
        "detector": {
            k: v
            for k, v in bundle.detector.model_dump(mode="json").items()
            if k
            in (
                "model_size",
                "checkpoint_sha256",
                "input_height",
                "input_width",
                "score_threshold",
                "runtime",
                "onnx_sha256",
                "training_run",
                "epoch",
            )
        },
        "preprocessing_version": bundle.preprocessing_version,
        "tracker": bundle.tracker.model_dump(mode="json"),
        "counting_policy": bundle.counting_policy,
        "calibration_version": bundle.calibration_version,
    }


def write_manifest(manifest: ReleaseManifest, path: Path) -> None:
    """Write ``manifest``; an existing one may only be rewritten unchanged (except the
    image digest, which a release workflow fills in once)."""
    data = manifest.model_dump(mode="json")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in existing.items() if k not in ("created_at", "image_digest")}
        mine = {k: v for k, v in data.items() if k not in ("created_at", "image_digest")}
        if canonical_json(comparable) != canonical_json(mine):
            raise ManifestExistsError(f"{path} exists with different content; use a new release")
        if existing.get("image_digest") not in (None, data["image_digest"]):
            raise ManifestExistsError(f"{path} already records image {existing['image_digest']}")
        data["created_at"] = existing["created_at"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_manifest(path: Path) -> ReleaseManifest:
    return ReleaseManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def check_model_info(manifest: ReleaseManifest, info: dict[str, Any]) -> list[str]:
    """Differences between a service's ``/v1/model-info`` and the manifest (empty = match)."""
    expected = manifest.bundle
    actual: dict[str, Any] = {
        "pipeline_config_sha256": info.get("pipeline_config_sha256"),
        "detector": {k: info.get("detector", {}).get(k) for k in expected["detector"]},
        "preprocessing_version": info.get("preprocessing_version"),
        "tracker": info.get("tracker"),
        "counting_policy": info.get("counting_policy"),
        "calibration_version": info.get("calibration_version"),
    }
    problems = []
    if info.get("pipeline_version") != manifest.release:
        problems.append(f"pipeline_version: {info.get('pipeline_version')} != {manifest.release}")
    for key, want in expected.items():
        got = actual[key]
        if key == "detector":
            for field, value in want.items():
                if got.get(field) != value:
                    problems.append(f"detector.{field}: {got.get(field)} != {value}")
        elif got != want:
            problems.append(f"{key}: {got} != {want}")
    return problems
