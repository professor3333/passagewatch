"""Release bundles: the versioned inference configuration that the service runs.

A bundle is a directory with ``bundle.json`` and the detector checkpoint it names. The API
reads only ``bundle.json`` (to register the pipeline version and answer ``/v1/model-info``);
the worker also loads the weights, after checking their SHA-256. ``bundle.json`` describes
everything that determines results, and its canonical hash is the pipeline config hash used
in the result cache, so any change to a counted result's inputs gives a new cache key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from passagewatch.service.jobs import canonical_json, sha256_text

BUNDLE_FILE = "bundle.json"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DetectorSpec(_Frozen):
    kind: Literal["yolox"] = "yolox"
    model_size: Literal["tiny", "s"]
    checkpoint: str  # file name inside the bundle directory
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_height: int = Field(gt=0)
    input_width: int = Field(gt=0)
    score_threshold: float = Field(gt=0, lt=1)
    training_run: str
    epoch: int = Field(ge=1)


class TrackerSpec(_Frozen):
    kind: Literal["kalman"] = "kalman"
    config: dict[str, Any]


class ReleaseBundle(_Frozen):
    pipeline_version: str = Field(pattern=r"^[a-z0-9][a-z0-9.+-]*$")
    detector: DetectorSpec
    preprocessing_version: str
    tracker: TrackerSpec
    counting_policy: str
    calibration_version: str | None = None
    provenance: dict[str, Any] = {}

    def config(self) -> dict[str, Any]:
        """Everything that determines a result, for the pipeline config hash."""
        return self.model_dump(mode="json", exclude={"provenance"})

    def config_sha256(self) -> str:
        return sha256_text(canonical_json(self.config()))


def load_bundle(bundle_dir: Path) -> ReleaseBundle:
    data = json.loads((bundle_dir / BUNDLE_FILE).read_text(encoding="utf-8"))
    return ReleaseBundle.model_validate(data)


class BundleExistsError(ValueError):
    """A bundle version already exists with different content."""


def build_bundle(
    *,
    checkpoint: Path,
    tracker_config: dict[str, Any],
    score_threshold: float,
    version: str,
    bundles_dir: Path,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Create ``bundles_dir/<version>/`` from a training checkpoint; versions are immutable.

    The checkpoint is copied as ``detector.pt`` and its SHA-256 recorded. The detector's size,
    input, training run and epoch are read from the checkpoint itself, not passed in.
    """
    import hashlib
    import shutil

    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    bundle = ReleaseBundle(
        pipeline_version=version,
        detector=DetectorSpec(
            model_size=config["model_size"],
            checkpoint="detector.pt",
            checkpoint_sha256=digest,
            input_height=config["input_height"],
            input_width=config["input_width"],
            score_threshold=score_threshold,
            training_run=config["name"],
            epoch=int(payload["state"]["epoch"]),
        ),
        preprocessing_version=payload["preprocessing_version"],
        tracker=TrackerSpec(config=tracker_config),
        counting_policy="cfc-compatible-v1",
        provenance={"training_metadata": payload.get("metadata", {}), **(provenance or {})},
    )
    target = bundles_dir / version
    if target.exists():
        existing = load_bundle(target)
        if existing.config() != bundle.config():
            raise BundleExistsError(
                f"bundle {version} exists with a different configuration; use a new version"
            )
        if existing.provenance != bundle.provenance:
            raise BundleExistsError(
                f"bundle {version} exists with the same configuration but different provenance "
                f"({existing.provenance}); rebuild with the same options or use a new version"
            )
        return target
    tmp = bundles_dir / f".{version}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copyfile(checkpoint, tmp / "detector.pt")
    (tmp / BUNDLE_FILE).write_text(
        json.dumps(bundle.model_dump(mode="json"), indent=1) + "\n", encoding="utf-8"
    )
    tmp.rename(target)
    return target


def activate(bundles_dir: Path, version: str) -> Path:
    """Point ``bundles_dir/active`` at a version (the active-release pointer)."""
    if not (bundles_dir / version / BUNDLE_FILE).is_file():
        raise FileNotFoundError(f"no bundle {version} in {bundles_dir}")
    link = bundles_dir / "active"
    tmp = bundles_dir / ".active.tmp"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(version)
    tmp.replace(link)
    return link
