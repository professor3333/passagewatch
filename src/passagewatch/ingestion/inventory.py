"""SHA-256 inventory of extracted files.

The inventory records every file's relative path, size, and SHA-256 so that later stages
can detect changed, missing, or duplicated files independently of the publisher's archive
checksums.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from passagewatch.ingestion.extract import EXTRACTED_MARKER

CHUNK_SIZE = 1 << 20

_INVENTORY_FIELDS: list[pa.Field[Any]] = [
    pa.field("path", pa.string(), nullable=False),
    pa.field("size", pa.int64(), nullable=False),
    pa.field("sha256", pa.string(), nullable=False),
]
INVENTORY_SCHEMA = pa.schema(_INVENTORY_FIELDS)


@dataclass(frozen=True)
class InventoryEntry:
    path: str
    size: int
    sha256: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name == EXTRACTED_MARKER and Path(dirpath) == root:
                continue
            yield Path(dirpath) / name


def build_inventory(root: Path) -> list[InventoryEntry]:
    """Hash every file under ``root``; paths are POSIX-style and relative to ``root``."""
    entries = [
        InventoryEntry(
            path=path.relative_to(root).as_posix(),
            size=path.stat().st_size,
            sha256=sha256_of(path),
        )
        for path in _iter_files(root)
    ]
    entries.sort(key=lambda e: e.path)
    return entries


def write_inventory(entries: list[InventoryEntry], out_path: Path) -> None:
    table = pa.Table.from_pylist(
        [{"path": e.path, "size": e.size, "sha256": e.sha256} for e in entries],
        schema=INVENTORY_SCHEMA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out_path)


def read_inventory(path: Path) -> list[InventoryEntry]:
    table = pq.read_table(path, schema=INVENTORY_SCHEMA)
    return [InventoryEntry(**row) for row in table.to_pylist()]
