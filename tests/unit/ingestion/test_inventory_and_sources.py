from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from passagewatch.ingestion.extract import EXTRACTED_MARKER
from passagewatch.ingestion.inventory import build_inventory, read_inventory, write_inventory
from passagewatch.ingestion.sources import SourceRegistry, load_registry

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_inventory_hashes_files_in_sorted_relative_order(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "2.jpg").write_bytes(b"two")
    (tmp_path / "a.txt").write_bytes(b"one")
    (tmp_path / EXTRACTED_MARKER).write_text("{}")

    entries = build_inventory(tmp_path)

    assert [e.path for e in entries] == ["a.txt", "b/2.jpg"]
    assert entries[1].size == 3
    assert entries[1].sha256 == hashlib.sha256(b"two").hexdigest()


def test_inventory_round_trips_through_parquet(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "f.bin").write_bytes(b"\x00\x01")
    entries = build_inventory(tmp_path / "data")

    write_inventory(entries, tmp_path / "inv" / "data.parquet")

    assert read_inventory(tmp_path / "inv" / "data.parquet") == entries


def test_repository_sources_config_is_valid() -> None:
    registry = load_registry(REPO_ROOT / "configs/data/cfc_sources.yaml")

    tiny = registry.bundle_files("tiny")

    assert registry.license == "MIT"
    assert {f.key for f in tiny} >= {"tiny_dataset.tar.gz", "fish_counting_annotations.tar.gz"}
    assert all(f.url.startswith("https://data.caltech.edu/records/") for f in tiny)
    assert registry.bundles["tiny"].contains_test_locations
    assert not registry.bundles["kenai-trainval"].contains_test_locations


def test_bundle_referencing_unknown_file_is_rejected() -> None:
    config = {
        "dataset": "d",
        "title": "t",
        "license": "MIT",
        "homepage": "h",
        "citation": "c",
        "url_template": "https://x/{record}/{key}",
        "records": [{"id": "r", "version": "1", "files": []}],
        "bundles": {
            "b": {"description": "d", "contains_test_locations": False, "files": ["r/missing"]}
        },
    }

    with pytest.raises(ValidationError, match="unknown files"):
        SourceRegistry.model_validate(config)


def test_malformed_checksum_is_rejected() -> None:
    config = {
        "dataset": "d",
        "title": "t",
        "license": "MIT",
        "homepage": "h",
        "citation": "c",
        "url_template": "https://x/{record}/{key}",
        "records": [{"id": "r", "version": "1", "files": [{"key": "k", "size": 1, "md5": "xyz"}]}],
        "bundles": {},
    }

    with pytest.raises(ValidationError):
        SourceRegistry.model_validate(config)
