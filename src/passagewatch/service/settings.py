"""Service settings, read from ``PASSAGEWATCH_*`` environment variables.

Every limit is explicit, so a deployment can see exactly what it accepts: upload size, frame
count, queue length, retries, deadline, and how long uploads are kept.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

PREFIX = "PASSAGEWATCH_"


class ServiceSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data_dir: Path = Path("var")
    bundle_dir: Path = Path("bundles/active")
    max_upload_bytes: int = Field(default=500 * 1024 * 1024, gt=0)
    max_frames: int = Field(default=6000, gt=0)
    upload_retention_hours: float = Field(default=24.0, gt=0)
    max_queue: int = Field(default=20, gt=0)
    max_attempts: int = Field(default=3, gt=0)
    job_deadline_seconds: int = Field(default=3600, gt=0)
    lease_seconds: int = Field(default=120, gt=0)
    tracks_page_limit: int = Field(default=200, gt=0)
    # Worker
    device: str = "cpu"  # the serving target is CPU; "auto" also tries CUDA, then MPS
    inference_batch: int = Field(default=8, gt=0)
    worker_poll_seconds: float = Field(default=2.0, gt=0)
    heartbeat_seconds: float = Field(default=20.0, gt=0)
    retention_sweep_seconds: float = Field(default=600.0, gt=0)
    worker_stale_seconds: float = Field(default=60.0, gt=0)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "service.db"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> ServiceSettings:
        env = os.environ if environ is None else environ
        values = {
            name: env[PREFIX + name.upper()]
            for name in cls.model_fields
            if PREFIX + name.upper() in env
        }
        return cls.model_validate(values)
