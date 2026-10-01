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
