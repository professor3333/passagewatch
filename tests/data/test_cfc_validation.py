from __future__ import annotations

import json
from pathlib import Path

import pytest

from passagewatch.ingestion.cfc import CfcLayout, load_clip
from passagewatch.validation.cfc import ClipReport, Code, ValidationReport, validate_dataset

from .conftest import LOCATION, NUM_FRAMES, WIDTH, FakeCfc, mot_line

REPO_ROOT = Path(__file__).resolve().parents[2]

# A fish moving right through the tiny window [10, 20); frames are 1-based in gt.txt.
GOOD_ROWS = [mot_line(f, 1, 1 + f, 20, 5, 4) for f in range(8, 24)]


def validate(fake: FakeCfc, subset: str = "tiny") -> ValidationReport:
    layout = CfcLayout.tiny(fake.root) if subset == "tiny" else CfcLayout.full(fake.root)
    return validate_dataset(layout, locations=[LOCATION])


def only_clip(report: ValidationReport) -> ClipReport:
    assert len(report.clips) == 1
    return report.clips[0]


def codes(clip: ClipReport) -> set[Code]:
    return {i.code for i in clip.issues}


def test_clean_clip_passes_with_window_stats(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip("clean", GOOD_ROWS)

    clip = only_clip(validate(fake_cfc))

    assert clip.status == "ok", clip.issues
    assert clip.stats is not None
    assert (clip.stats.frame_start, clip.stats.frame_stop) == (10, 20)
    # gt frames 11..20 are images 10..19.
    assert clip.stats.boxes == 10
    assert clip.stats.tracks_cut_by_window == 1
    assert clip.stats.frames_checked == 10
    assert clip.stats.image_sizes == ((40, 60),)


def test_tiny_clip_takes_boxes_from_full_gt_not_gt_tiny(fake_cfc: FakeCfc) -> None:
    # Mimic CFC's tool: gt_tiny keeps the *first* annotated rows, misaligned with the images.
    fake_cfc.add_clip("tiny", GOOD_ROWS, tiny_rows=GOOD_ROWS[:3])
    layout = CfcLayout.tiny(fake_cfc.root)
    meta = layout.metadata(LOCATION).clips["tiny"]

    clip = load_clip(layout, LOCATION, meta)

    assert only_clip(validate(fake_cfc)).status == "ok"
    assert clip.annotations.frame_index.tolist() == list(range(10, 20))
    assert clip.num_window_frames == 10


def test_gap_in_frames_quarantines_the_clip(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip("gappy", GOOD_ROWS, window=[10, 11, 12, 15, 16])

    clip = only_clip(validate(fake_cfc))

    assert clip.status == "quarantined"
    assert clip.quarantine_reasons == ["missing_frames"]
    assert clip.issues[0].examples == ("13-14",)


@pytest.mark.parametrize(
    ("row", "code"),
    [
        (mot_line(NUM_FRAMES + 1, 2, 5, 5, 5, 5), Code.FRAME_OUT_OF_RANGE),
        (mot_line(0, 2, 5, 5, 5, 5), Code.FRAME_OUT_OF_RANGE),
        (mot_line(12, 1, 5, 5, 5, 5), Code.DUPLICATE_TRACK_FRAME),
        (mot_line(12, 2, 5, 5, 0, 5), Code.NON_POSITIVE_BOX_SIZE),
        (mot_line(12, 2, WIDTH + 1, 5, 5, 5), Code.BOX_OUTSIDE_IMAGE),
    ],
)
def test_invalid_annotations_quarantine_the_clip(fake_cfc: FakeCfc, row: str, code: Code) -> None:
    fake_cfc.add_clip("bad", [*GOOD_ROWS, row])

    clip = only_clip(validate(fake_cfc))

    assert clip.status == "quarantined"
    assert clip.quarantine_reasons == [code.value]


def test_frame_out_of_range_is_caught_outside_the_tiny_window(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip("late", [*GOOD_ROWS, mot_line(NUM_FRAMES + 5, 3, 5, 5, 5, 5)])

    assert only_clip(validate(fake_cfc)).quarantine_reasons == ["frame_out_of_range"]


def test_malformed_gt_quarantines_the_clip(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip("broken", [*GOOD_ROWS, "12,1,5,5\n"])

    clip = only_clip(validate(fake_cfc))

    assert clip.quarantine_reasons == ["mot_format"]
    assert "gt.txt:17" in clip.issues[0].message


def test_box_partly_outside_is_a_warning_and_kept_unclipped(fake_cfc: FakeCfc) -> None:
    # bb_top = -1 occurs in CFC: internal y_min = -2.
    fake_cfc.add_clip("edge", [*GOOD_ROWS, mot_line(12, 2, 5, -1, 5, 5)])
    layout = CfcLayout.tiny(fake_cfc.root)

    clip = only_clip(validate(fake_cfc))
    loaded = load_clip(layout, LOCATION, layout.metadata(LOCATION).clips["edge"])

    assert clip.status == "warning"
    assert codes(clip) == {Code.BOX_PARTIALLY_OUTSIDE_IMAGE}
    assert loaded.annotations.boxes[:, 1].min() == -2.0


@pytest.mark.parametrize(
    ("size", "status", "code"),
    [
        ((WIDTH, 61), "warning", Code.IMAGE_SIZE_DIFFERS_SLIGHTLY),
        ((WIDTH + 3, 60), "quarantined", Code.IMAGE_SIZE_MISMATCH),
    ],
)
def test_image_size_is_checked_against_metadata(
    fake_cfc: FakeCfc, size: tuple[int, int], status: str, code: Code
) -> None:
    fake_cfc.add_clip("sized", GOOD_ROWS, image_size=size)

    clip = only_clip(validate(fake_cfc))

    assert clip.status == status
    assert codes(clip) == {code}
    assert clip.issues[0].count == 10


def test_undecodable_frame_quarantines_the_clip(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip("corrupt", GOOD_ROWS)
    (fake_cfc.tiny / "raw" / LOCATION / "corrupt" / "13.jpg").write_bytes(b"not a jpeg")

    clip = only_clip(validate(fake_cfc))

    assert clip.quarantine_reasons == ["unreadable_frame"]
    assert clip.issues[0].examples == ("13",)


def test_tiny_rows_must_come_from_gt(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip("drift", GOOD_ROWS, tiny_rows=[mot_line(12, 1, 9, 9, 5, 5)])

    assert only_clip(validate(fake_cfc)).quarantine_reasons == ["tiny_rows_not_in_gt"]


def test_stray_files_are_recorded_as_warnings(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip("notebook", GOOD_ROWS)
    (fake_cfc.annotations / LOCATION / "notebook" / ".ipynb_checkpoints").mkdir()
    (fake_cfc.tiny / "raw" / LOCATION / "notebook" / "Thumbs.db").write_bytes(b"")

    clip = only_clip(validate(fake_cfc))

    assert clip.status == "warning"
    assert [i.examples for i in clip.issues] == [(".ipynb_checkpoints",), ("Thumbs.db",)]


def test_clip_and_metadata_must_match(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip("present", GOOD_ROWS)
    entry = json.loads((fake_cfc.tiny / f"metadata-tiny/{LOCATION}.json").read_text())[0]
    fake_cfc.add_metadata({**entry, "clip_name": "ghost"})
    (fake_cfc.tiny / "raw" / LOCATION / "orphan").mkdir()

    report = validate(fake_cfc)

    by_name = {c.clip_name: c for c in report.clips}
    assert by_name["ghost"].quarantine_reasons == ["missing_clip_directory"]
    assert by_name["orphan"].quarantine_reasons == ["missing_metadata"]
    assert by_name["present"].status == "ok"


def test_missing_metadata_file_is_a_location_issue(fake_cfc: FakeCfc) -> None:
    fake_cfc.root.mkdir()

    report = validate(fake_cfc)

    assert report.clips == []
    assert report.location_issues[LOCATION][0].code is Code.MISSING_METADATA_FILE


def test_full_layout_without_frames_validates_annotations_only(fake_cfc: FakeCfc) -> None:
    fake_cfc.add_clip("full", GOOD_ROWS, window=None)

    report = validate(fake_cfc, subset="full")

    clip = only_clip(report)
    assert clip.status == "ok"
    assert clip.stats is not None
    assert (clip.stats.frame_start, clip.stats.frame_stop) == (0, NUM_FRAMES)
    assert clip.stats.boxes == len(GOOD_ROWS)
    assert not report.images_checked


def test_report_json_is_valid_and_summarized(fake_cfc: FakeCfc, tmp_path: Path) -> None:
    fake_cfc.add_clip("clean", GOOD_ROWS)
    fake_cfc.add_clip("gappy", GOOD_ROWS, window=[10, 12])
    report = validate(fake_cfc)

    report.write_json(tmp_path / "report.json")
    data = json.loads((tmp_path / "report.json").read_text())

    assert data["summary"]["quarantined"] == 1
    assert data["summary"]["by_location"][LOCATION] == {"ok": 1, "warning": 0, "quarantined": 1}
    assert [c["status"] for c in data["clips"]] == ["ok", "quarantined"]
    assert data["clips"][1]["quarantine_reasons"] == ["missing_frames"]


TINY_ROOT = REPO_ROOT / "data/extracted/cfc"


@pytest.mark.slow
@pytest.mark.skipif(
    not (TINY_ROOT / "tiny_dataset/tiny_dataset").is_dir(), reason="CFC tiny subset not present"
)
def test_real_tiny_subset_has_no_quarantined_clips() -> None:
    report = validate_dataset(CfcLayout.tiny(TINY_ROOT))

    assert len(report.clips) == 120
    assert report.quarantined == []
    assert report.location_issues == {}
