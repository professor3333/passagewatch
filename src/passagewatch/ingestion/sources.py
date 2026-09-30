"""Typed registry of publisher files, loaded from ``configs/data/*.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MD5_PATTERN = r"^[0-9a-f]{32}$"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceFile(_Frozen):
    key: str = Field(min_length=1)
    size: int = Field(gt=0)
    md5: str = Field(pattern=MD5_PATTERN)


class Record(_Frozen):
    id: str = Field(min_length=1)
    version: str
    files: tuple[SourceFile, ...]


class Bundle(_Frozen):
    description: str
    contains_test_locations: bool
    files: tuple[str, ...] = Field(min_length=1)


class ResolvedFile(_Frozen):
    """A publisher file with everything needed to fetch and verify it."""

    record: str
    key: str
    size: int
    md5: str
    url: str

    @property
    def ref(self) -> str:
        return f"{self.record}/{self.key}"


class SourceRegistry(_Frozen):
    dataset: str
    title: str
    license: str
    homepage: str
    citation: str
    url_template: str
    records: tuple[Record, ...]
    bundles: dict[str, Bundle]

    @model_validator(mode="after")
    def _bundles_reference_known_files(self) -> Self:
        known = {f"{r.id}/{f.key}" for r in self.records for f in r.files}
        for name, bundle in self.bundles.items():
            unknown = [ref for ref in bundle.files if ref not in known]
            if unknown:
                raise ValueError(f"bundle {name!r} references unknown files: {unknown}")
        return self

    def resolve(self, ref: str) -> ResolvedFile:
        record_id, _, key = ref.partition("/")
        for record in self.records:
            if record.id != record_id:
                continue
            for source in record.files:
                if source.key == key:
                    return ResolvedFile(
                        record=record_id,
                        key=key,
                        size=source.size,
                        md5=source.md5,
                        url=self.url_template.format(record=record_id, key=key),
                    )
        raise KeyError(f"unknown source file: {ref}")

    def bundle_files(self, name: str) -> list[ResolvedFile]:
        if name not in self.bundles:
            raise KeyError(f"unknown bundle {name!r}; available: {sorted(self.bundles)}")
        return [self.resolve(ref) for ref in self.bundles[name].files]


def load_registry(path: Path) -> SourceRegistry:
    with path.open("r", encoding="utf-8") as fh:
        return SourceRegistry.model_validate(yaml.safe_load(fh))
