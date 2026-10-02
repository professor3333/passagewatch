from __future__ import annotations

import dataclasses
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from passagewatch.ingestion.cfc import CfcLayout
from passagewatch.ingestion.manifest import (
    MANIFEST_SCHEMA,
    ManifestVersionError,
    build_rows,
    read_manifest,
    write_manifest,
)
from passagewatch.validation.cfc import ValidationReport, merge_frame_reports, validate_dataset

from .conftest import LOCATION, FakeCfc, clip_name, mot_line

ROWS = [mot_line(f, 1, 1 + f, 20, 5, 4) for f in range(8, 24)]
INPUTS = {"validation_report": "0" * 64}


def build(fake: FakeCfc, locations: list[str]) -> list[dict[str, object]]:
    layout = CfcLayout.tiny(fake.root)
    return build_rows(layout, validate_dataset(layout, locations=locations))


def test_rows_record_partition_recording_and_status(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip(clip_name("good"), ROWS)
    fake_cfc.add_clip(clip_name("gappy"), ROWS, window=[10, 12])
    fake_cfc.add_clip(clip_name("test", 500), ROWS, location="elwha")

    rows = build(fake_cfc, [LOCATION, "elwha"])

    by_name = {r["clip_name"]: r for r in rows}
    good = by_name[clip_name("good")]
    assert good["partition"] == "train"
    assert good["tuning_allowed"] is True
    assert good["recording_id"] == "good_2018-06-01_120000"
    assert (good["recording_frame_start"], good["recording_frame_stop"]) == (100, 130)
    assert (good["frame_start"], good["frame_stop"], good["boxes"]) == (10, 20, 10)
    assert good["usable"] and good["frames_validated"]
    assert len(str(good["gt_sha256"])) == 64

    gappy = by_name[clip_name("gappy")]
    assert gappy["status"] == "quarantined"
    assert not gappy["usable"]
    assert gappy["quarantine_reasons"] == ["missing_frames"]

    test_clip = by_name[clip_name("test", 500)]
    assert test_clip["partition"] == "test"
    assert test_clip["tuning_allowed"] is False
    # Ordered by the canonical location order, then clip name.
    assert [r["location"] for r in rows] == [LOCATION, LOCATION, "elwha"]


def test_manifest_round_trips_and_matches_schema(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    fake_cfc.add_clip(clip_name("good"), ROWS)
    rows = build(fake_cfc, [LOCATION])

    paths = write_manifest(rows, tmp_path, "tiny-v1", inputs=INPUTS)

    assert pq.read_schema(paths.parquet).equals(MANIFEST_SCHEMA)
    assert read_manifest(paths.parquet) == rows


def test_existing_version_is_immutable(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    fake_cfc.add_clip(clip_name("good"), ROWS)
    rows = build(fake_cfc, [LOCATION])
    paths = write_manifest(rows, tmp_path, "tiny-v1", inputs=INPUTS)
    before = paths.parquet.stat().st_mtime_ns

    # Identical content: a no-op.
    write_manifest(rows, tmp_path, "tiny-v1", inputs=INPUTS)
    assert paths.parquet.stat().st_mtime_ns == before

    changed = [{**rows[0], "boxes": 999}]
    with pytest.raises(ManifestVersionError, match="new version"):
        write_manifest(changed, tmp_path, "tiny-v1", inputs=INPUTS)
    with pytest.raises(ManifestVersionError, match="new version"):
        write_manifest(rows, tmp_path, "tiny-v1", inputs={"validation_report": "1" * 64})

    write_manifest(changed, tmp_path, "tiny-v2", inputs=INPUTS)


def test_tampered_manifest_fails_its_hash_check(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    fake_cfc.add_clip(clip_name("good"), ROWS)
    rows = build(fake_cfc, [LOCATION])
    paths = write_manifest(rows, tmp_path, "tiny-v1", inputs=INPUTS)
    table = pq.read_table(paths.parquet).to_pylist()
    table[0]["partition"] = "val"
    pq.write_table(pq.read_table(paths.parquet).from_pylist(table, MANIFEST_SCHEMA), paths.parquet)

    with pytest.raises(ManifestVersionError, match="does not match"):
        read_manifest(paths.parquet)


@pytest.mark.parametrize("version", ["v1", "tiny", "tiny-v0", "kenai-v1", "tiny-v1.parquet"])
def test_version_names_are_checked(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError, match="manifest version"):
        write_manifest([], tmp_path, version, inputs=INPUTS)


def test_holdout_clips_get_their_own_partition(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip(clip_name("kept"), ROWS)
    fake_cfc.add_clip(clip_name("held"), ROWS)
    layout = CfcLayout.tiny(fake_cfc.root)
    report = validate_dataset(layout, locations=[LOCATION])

    rows = {r["clip_name"]: r for r in build_rows(layout, report, frozenset({clip_name("held")}))}

    held, kept = rows[clip_name("held")], rows[clip_name("kept")]
    assert (held["partition"], held["official_split"], held["tuning_allowed"]) == (
        "holdout",
        "train",
        False,
    )
    assert (kept["partition"], kept["tuning_allowed"]) == ("train", True)


def test_only_train_clips_can_be_held_out(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip(clip_name("test", 500), ROWS, location="elwha")
    layout = CfcLayout.tiny(fake_cfc.root)
    report = validate_dataset(layout, locations=["elwha"])

    with pytest.raises(ValueError, match="not a kenai-train clip"):
        build_rows(layout, report, frozenset({clip_name("test", 500)}))


def test_frame_reports_merge_by_the_subset_that_checked_each_clip(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip(clip_name("a"), ROWS)
    fake_cfc.add_clip(clip_name("b"), ROWS)
    a, b = clip_name("a"), clip_name("b")
    frames = CfcLayout.tiny(fake_cfc.root).frames_dir
    annotations = CfcLayout.tiny(fake_cfc.root).annotations_dir

    def checked(only: str) -> ValidationReport:
        layout = dataclasses.replace(
            CfcLayout.tiny(fake_cfc.root), frames_dir=frames, frame_clips=frozenset({only})
        )
        assert layout.annotations_dir == annotations
        return validate_dataset(layout, locations=[LOCATION])

    merged = merge_frame_reports([(checked(a), frozenset({a})), (checked(b), frozenset({b}))])

    stats = {c.clip_name: c.stats for c in merged.clips}
    assert stats[a] is not None and stats[a].frames_checked > 0
    assert stats[b] is not None and stats[b].frames_checked > 0
    with pytest.raises(ValueError, match="more than one frames subset"):
        merge_frame_reports([(checked(a), frozenset({a})), (checked(a), frozenset({a}))])
