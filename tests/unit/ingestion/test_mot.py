from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from passagewatch.ingestion.mot import (
    BoxAnnotations,
    MotFormatError,
    internal_to_mot,
    mot_to_internal,
    read_mot,
    read_mot_rows,
    write_mot,
)

# Real rows from CFC elwha gt.txt: 1-based frame and box origin.
CFC_TEXT = (
    "209,1,76.000,328.000,23.000,9.000,1.0,-1,-1,-1\n"
    "210,1,73.000,315.000,31.000,14.000,1.0,-1,-1,-1\n"
    "211,2,1.000,1.000,32.500,15.250,1.0,-1,-1,-1\n"
)


def test_one_based_mot_converts_to_zero_based_internal() -> None:
    rows = np.array([[1, 7, 1, 1, 10, 5], [451, 7, 76, 328, 23, 9]], dtype=np.float64)

    ann = mot_to_internal(rows)

    # gt frame N is image N-1.jpg; the image's top-left pixel (1, 1) becomes (0, 0).
    assert ann.frame_index.tolist() == [0, 450]
    assert ann.track_id.tolist() == [7, 7]
    assert ann.boxes.tolist() == [[0, 0, 10, 5], [75, 327, 98, 336]]


def test_conversion_round_trips_exactly() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    rows = np.column_stack(
        [
            rng.integers(1, 600, n),
            rng.integers(1, 50, n),
            rng.integers(-1, 2000, n) + rng.integers(0, 1000, n) / 1000,
            rng.integers(-1, 2000, n) + rng.integers(0, 1000, n) / 1000,
            rng.integers(1, 200, n) + rng.integers(0, 1000, n) / 1000,
            rng.integers(1, 200, n) + rng.integers(0, 1000, n) / 1000,
        ]
    ).astype(np.float64)
    rows = np.round(rows, 3)

    back = internal_to_mot(mot_to_internal(rows))

    np.testing.assert_array_equal(back[:, :2], rows[:, :2])
    # Box values carry 3 decimals in CFC; the round trip is exact at that precision.
    np.testing.assert_array_equal(np.round(back[:, 2:], 3), rows[:, 2:])


def test_file_round_trip_reproduces_cfc_text(tmp_path: Path) -> None:
    source = tmp_path / "gt.txt"
    source.write_text(CFC_TEXT)

    ann = read_mot(source)
    write_mot(ann, tmp_path / "out" / "gt.txt")

    assert (tmp_path / "out" / "gt.txt").read_text() == CFC_TEXT
    assert read_mot(tmp_path / "out" / "gt.txt").equals(ann)


def test_blank_lines_and_empty_files_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "gt.txt"
    path.write_text("\n" + CFC_TEXT + "\n\n")
    assert len(read_mot(path)) == 3

    path.write_text("")
    assert len(read_mot(path)) == 0


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("1,1,1,1,5,5,1.0,-1,-1", "expected 10 columns, got 9"),
        ("1.5,1,1,1,5,5,1.0,-1,-1,-1", "not an integer"),
        ("1,x,1,1,5,5,1.0,-1,-1,-1", "could not convert"),
        ("1,1,nan,1,5,5,1.0,-1,-1,-1", "non-finite"),
    ],
)
def test_malformed_rows_report_file_and_line(tmp_path: Path, line: str, reason: str) -> None:
    path = tmp_path / "gt.txt"
    path.write_text(CFC_TEXT + line + "\n")

    with pytest.raises(MotFormatError, match=reason) as info:
        read_mot_rows(path)

    assert info.value.line_number == 4
    assert info.value.path == path


def test_integral_float_frame_numbers_are_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "gt.txt"
    path.write_text("12.0,3,1,1,5,5,1.0,-1,-1,-1\n")

    assert read_mot(path).frame_index.tolist() == [11]


def test_frame_range_selection_is_half_open() -> None:
    rows = np.array([[f, 1, 1, 1, 2, 2] for f in range(1, 11)], dtype=np.float64)
    ann = mot_to_internal(rows)

    assert ann.in_frame_range(3, 6).frame_index.tolist() == [3, 4, 5]


def test_inconsistent_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistent shapes"):
        BoxAnnotations(
            frame_index=np.zeros(2, dtype=np.int64),
            track_id=np.zeros(1, dtype=np.int64),
            boxes=np.zeros((2, 4)),
        )
