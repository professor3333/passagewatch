from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from passagewatch.inference.batch import (
    make_layout,
    make_tasks,
    run_tasks,
    select_rows,
    summarize_outcomes,
)
from passagewatch.inference.classical import ClassicalConfig
from passagewatch.ingestion.cfc import CfcLayout
from passagewatch.ingestion.manifest import build_rows, write_manifest
from passagewatch.ingestion.mot import read_mot
from passagewatch.validation.cfc import validate_dataset

from .conftest import LOCATION, NUM_FRAMES, FakeCfc, clip_name, mot_line

# A 6 x 3 m "sonar" frame at 0.2 m per pixel ... rendered at 1 cm per pixel below.
WIDTH, HEIGHT = 40, 60


def add_passage_clip(fake: FakeCfc, label: str) -> str:
    """A bright target crossing left to right across frames 10..19, annotated (1-based)."""
    name = clip_name(label)
    rows = [mot_line(f + 1, 1, 1 + 3 * (f - 10), 30, 8, 4) for f in range(10, 20)]
    fake.add_clip(
        name,
        rows,
        metadata={
            "x_meter_start": 0.0,
            "x_meter_stop": 0.4,
            "y_meter_start": 0.6,
            "y_meter_stop": 0.0,
        },
    )
    frames_dir = fake.tiny / "raw" / LOCATION / name
    rng = np.random.default_rng(0)
    for f in range(10, 20):
        image = np.clip(rng.normal(60, 4, (HEIGHT, WIDTH)), 0, 255).astype(np.uint8)
        x = 3 * (f - 10)
        image[30:34, x : x + 8] = 200
        cv2.imwrite(str(frames_dir / f"{f}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return name


@pytest.fixture
def manifest(fake_cfc: FakeCfc, tmp_path: Path) -> Path:
    add_passage_clip(fake_cfc, "one")
    add_passage_clip(fake_cfc, "two")
    layout = CfcLayout.tiny(fake_cfc.root)
    rows = build_rows(layout, validate_dataset(layout, locations=[LOCATION]))
    paths = write_manifest(rows, tmp_path / "manifests", "tiny-v1", inputs={"x": "0" * 64})
    return paths.parquet


CONFIG = ClassicalConfig.model_validate(
    {
        "name": "test",
        "detector": {"background_window": 9, "min_area_m2": 0.0005, "max_area_m2": 0.05},
        "tracker": {"gate_m": 0.2, "min_length": 3, "min_hits": 3},
    }
)


def test_batch_runs_clips_and_writes_mot_tracks(
    fake_cfc: FakeCfc, manifest: Path, tmp_path: Path
) -> None:
    rows = select_rows(manifest, ["train"])
    layout = make_layout("tiny", fake_cfc.root, None, rows)

    outcomes = run_tasks(make_tasks(rows, layout, CONFIG, tmp_path / "out"), workers=1)
    summary = summarize_outcomes(outcomes)

    assert len(outcomes) == 2
    assert all(o.error.reference.right == 1 for o in outcomes)
    assert all(o.error.predicted.right == 1 for o in outcomes)
    assert summary["macro_nmae"] == 0.0
    assert summary["locations"][0]["detection_recall"] > 0.8
    written = read_mot(
        tmp_path / "out" / LOCATION / "classical" / "data" / f"{rows[0]['clip_name']}.txt"
    )
    assert written.frame_index.min() >= 10 and written.frame_index.max() < 20
    assert NUM_FRAMES == 30


def test_parallel_and_serial_runs_agree(fake_cfc: FakeCfc, manifest: Path) -> None:
    rows = select_rows(manifest, ["train"])
    layout = make_layout("tiny", fake_cfc.root, None, rows)
    tasks = make_tasks(rows, layout, CONFIG, None)

    serial = summarize_outcomes(run_tasks(tasks, workers=1))
    parallel = summarize_outcomes(run_tasks(tasks, workers=2))

    assert serial["locations"] == parallel["locations"]


def test_test_partition_is_refused(manifest: Path) -> None:
    with pytest.raises(ValueError, match="test partition"):
        select_rows(manifest, ["val", "test"])


def test_full_manifests_need_a_frames_directory(fake_cfc: FakeCfc) -> None:
    with pytest.raises(ValueError, match="frames-dir"):
        make_layout("full", fake_cfc.root, None, [])
