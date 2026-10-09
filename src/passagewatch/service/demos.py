"""Demo examples: CFC recordings shown as precomputed, visibly cached results (docs/demos.md).

The catalog (``demos/catalog.json``, committed) describes each example: what it shows, its
CFC source and reference counts, the sonar metadata for uploading it, and the SHA-256 of
its frames ZIP. The frames are not in git. ``scripts/package_demos.py`` builds the ZIPs
reproducibly into ``passagewatch-demos-<version>.tar.gz``, which is attached to a GitHub
release, and ``scripts/load_demos.py`` uploads them to a running service, where the active
release analyses each one once.

The service finds a demo by its recording's SHA-256: ``GET /v1/demos`` lists the catalog
with the clip and the precomputed job, if they are loaded. A demo's clip never expires and
cannot be deleted through the API. Visitors open a demo as their own job, a result-cache hit
of the precomputed one, so their corrections never change the shared example.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sqlite3
import tarfile
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from passagewatch.service.bundle_archive import NOTICES, sha256_file, write_tar_gz

ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # the earliest a ZIP can record: fixed, for reproducibility


class DemoMismatchError(ValueError):
    """A demo archive or ZIP that is not what the catalog describes."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DemoSource(_Frozen):
    dataset: Literal["Caltech Fish Counting (CFC) v1.1"]
    location: str
    clip_name: str
    # How the clip relates to training and selection, e.g. "kenai-holdout-v1".
    split_note: str


class DemoCounts(_Frozen):
    right: int = Field(ge=0)
    left: int = Field(ge=0)


class DemoMeters(_Frozen):
    x_start: float
    x_stop: float
    y_start: float
    y_stop: float


class DemoEntry(_Frozen):
    demo_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,40}$")
    kind: Literal["clear", "difficult", "unfamiliar-camera"]
    title: str
    summary: str  # what the example shows and what to look for
    source: DemoSource
    reference: DemoCounts  # CFC annotations counted with cfc-compatible-v1
    framerate: float = Field(gt=0)
    meters: DemoMeters
    num_frames: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # of the frames ZIP, as uploaded

    @property
    def zip_name(self) -> str:
        return f"{self.demo_id}.zip"


class DemoCatalog(_Frozen):
    version: str = Field(pattern=r"^demos-v[0-9]+$")
    demos: tuple[DemoEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> DemoCatalog:
        for field in ("demo_id", "sha256"):
            values = [getattr(d, field) for d in self.demos]
            if len(set(values)) != len(values):
                raise ValueError(f"demo {field}s must be unique")
        return self

    def by_sha256(self, sha256: str) -> DemoEntry | None:
        return next((d for d in self.demos if d.sha256 == sha256), None)

    @property
    def archive_name(self) -> str:
        return f"passagewatch-{self.version}.tar.gz"


def load_catalog(path: Path) -> DemoCatalog:
    return DemoCatalog.model_validate_json(path.read_text(encoding="utf-8"))


def frames_zip(frames: Sequence[Path]) -> bytes:
    """A ZIP of ``frames`` as ``0.jpg … N-1.jpg``, byte-identical for the same frames."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:  # JPEGs: no gain
        for index, path in enumerate(frames):
            info = zipfile.ZipInfo(f"{index}.jpg", date_time=ZIP_DATE)
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def pack_demos(catalog: DemoCatalog, zips_dir: Path, notices_dir: Path, out_dir: Path) -> Path:
    """Write ``out_dir/passagewatch-<version>.tar.gz`` from the catalog's ZIPs, after checking
    each one's SHA-256, with the catalog and the license notices; return its path."""
    for demo in catalog.demos:
        verify_zip(zips_dir / demo.zip_name, demo)
    prefix = f"passagewatch-{catalog.version}"
    members = [(f"{prefix}/{d.zip_name}", zips_dir / d.zip_name) for d in catalog.demos]
    members += [(f"{prefix}/{n}", notices_dir / n) for n in NOTICES]
    with tempfile.TemporaryDirectory() as tmp:
        catalog_file = Path(tmp) / "catalog.json"
        catalog_file.write_text(catalog_json(catalog), encoding="utf-8")
        members.append((f"{prefix}/catalog.json", catalog_file))
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / catalog.archive_name
        write_tar_gz(members, out)
    return out


def unpack_demos(archive: Path, catalog: DemoCatalog, dest: Path) -> dict[str, Path]:
    """Extract the catalog's ZIPs from ``archive`` into ``dest``, verified; ``{demo_id: zip}``.

    Only the ZIPs the catalog names are taken, and each must match its SHA-256.
    """
    prefix = f"passagewatch-{catalog.version}/"
    wanted = {prefix + d.zip_name: d for d in catalog.demos}
    dest.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar.getmembers():
            demo = wanted.get(member.name)
            if demo is None:
                continue
            if not member.isfile():
                raise DemoMismatchError(f"{member.name!r} is not a regular file")
            source = tar.extractfile(member)
            assert source is not None
            path = dest / demo.zip_name
            with source, path.open("wb") as fh:
                shutil.copyfileobj(source, fh)
            verify_zip(path, demo)
            found[demo.demo_id] = path
    missing = sorted(d.demo_id for d in catalog.demos if d.demo_id not in found)
    if missing:
        raise DemoMismatchError(f"{archive.name} lacks demos {missing}")
    return found


def verify_zip(path: Path, demo: DemoEntry) -> None:
    if not path.is_file():
        raise DemoMismatchError(f"{path} is missing")
    if sha256_file(path) != demo.sha256:
        raise DemoMismatchError(f"{path.name}: SHA-256 differs from the catalog")


def catalog_json(catalog: DemoCatalog) -> str:
    return json.dumps(catalog.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def demo_listing(
    conn: sqlite3.Connection, catalog: DemoCatalog, pipeline_version: str | None
) -> list[dict[str, Any]]:
    """Each demo with its loaded clip and the precomputed job of ``pipeline_version``.

    The precomputed job is the demo's first successful analysis by that pipeline that is
    not itself a cache hit. A demo with no such job is listed as not available.
    """
    listing = []
    for demo in catalog.demos:
        row = None
        if pipeline_version is not None:
            row = conn.execute(
                "SELECT c.clip_id, j.job_id, j.finished_at FROM clips c"
                " JOIN jobs j ON j.clip_id = c.clip_id"
                " WHERE c.sha256 = ? AND c.deleted_at IS NULL AND j.status = 'succeeded'"
                " AND j.cached_from IS NULL AND j.pipeline_version = ?"
                " ORDER BY j.finished_at, j.job_id LIMIT 1",
                (demo.sha256, pipeline_version),
            ).fetchone()
        listing.append(
            {
                "demo_id": demo.demo_id,
                "kind": demo.kind,
                "title": demo.title,
                "summary": demo.summary,
                "source": demo.source.model_dump(mode="json"),
                "reference": demo.reference.model_dump(mode="json"),
                "available": row is not None,
                "clip_id": None if row is None else row["clip_id"],
                "job_id": None if row is None else row["job_id"],
                "computed_at": None if row is None else row["finished_at"],
                "pipeline_version": None if row is None else pipeline_version,
            }
        )
    return listing
