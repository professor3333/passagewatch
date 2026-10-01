from __future__ import annotations

import json
from pathlib import Path

import pytest

from passagewatch.counting.policy import DirectionalCounts
from passagewatch.evaluation.nmae import (
    ClipCountError,
    GroupResult,
    evaluate_mot_location,
    macro_nmae,
    summarize,
)
from passagewatch.ingestion.metadata import ClipMetadata


def error(ref: tuple[int, int], pred: tuple[int, int], seconds: float = 60.0) -> ClipCountError:
    return ClipCountError("c", DirectionalCounts(*ref), DirectionalCounts(*pred), seconds)


def test_directions_are_never_netted() -> None:
    # 5 right + 5 left predicted as 10 right: the net is right, but the error is 10.
    assert error((5, 5), (10, 0)).absolute_error == 10


def test_group_nmae_sums_errors_and_passages_over_clips() -> None:
    result = summarize("loc", [error((3, 1), (2, 1)), error((0, 4), (1, 6))])

    assert (result.absolute_error, result.reference_passages) == (4, 8)
    assert result.nmae == pytest.approx(0.5)


def test_nmae_is_undefined_without_true_passages() -> None:
    result = summarize("empty", [error((0, 0), (2, 1), seconds=1800.0)])

    assert result.nmae is None
    assert result.absolute_error == 3
    assert result.predicted_passages_per_hour == pytest.approx(6.0)


def test_macro_average_skips_undefined_groups() -> None:
    groups = [
        GroupResult("a", 1, 1, 10, 10, 1.0),
        GroupResult("b", 1, 3, 10, 10, 1.0),
        GroupResult("c", 1, 2, 0, 2, 1.0),
    ]

    assert macro_nmae(groups) == pytest.approx(0.2)
    assert macro_nmae([groups[2]]) is None


def meta(name: str, num_frames: int = 20) -> ClipMetadata:
    return ClipMetadata(
        clip_name=name,
        num_frames=num_frames,
        framerate=10.0,
        width=100,
        height=50,
        x_meter_start=0,
        x_meter_stop=1,
        y_meter_start=1,
        y_meter_stop=0,
    )


def mot(rows: list[tuple[int, int, float]]) -> str:
    """1-based MOT rows for 10x10 boxes with left edge ``left`` (1-based)."""
    return "".join(f"{f},{t},{left:.3f},20.000,10.000,10.000,1.0,-1,-1,-1\n" for f, t, left in rows)


def write_clip(root: Path, name: str, gt: str, pred: str) -> None:
    (root / "gt" / name).mkdir(parents=True, exist_ok=True)
    (root / "gt" / name / "gt.txt").write_text(gt)
    (root / "pred").mkdir(exist_ok=True)
    (root / "pred" / f"{name}.txt").write_text(pred)


def test_mot_location_is_evaluated_per_clip(tmp_path: Path) -> None:
    rightward = mot([(1, 1, 11.0), (10, 1, 71.0)])
    leftward = mot([(1, 4, 71.0), (10, 4, 11.0)])
    write_clip(tmp_path, "a", rightward, rightward)
    write_clip(tmp_path, "b", rightward + leftward, leftward)
    write_clip(tmp_path, "c", "", "")

    errors = evaluate_mot_location(tmp_path / "gt", tmp_path / "pred", {n: meta(n) for n in "abc"})

    assert [(e.clip_name, e.absolute_error) for e in errors] == [("a", 0), ("b", 1), ("c", 0)]
    assert summarize("loc", errors).nmae == pytest.approx(1 / 3)


def test_missing_prediction_file_is_an_error(tmp_path: Path) -> None:
    write_clip(tmp_path, "a", "", "")

    with pytest.raises(FileNotFoundError, match="no prediction file for clip b"):
        evaluate_mot_location(tmp_path / "gt", tmp_path / "pred", {n: meta(n) for n in "ab"})


def test_prediction_outside_the_clip_is_an_error(tmp_path: Path) -> None:
    write_clip(tmp_path, "a", "", mot([(21, 1, 11.0)]))

    with pytest.raises(ValueError, match="outside frames"):
        evaluate_mot_location(tmp_path / "gt", tmp_path / "pred", {"a": meta("a")})


def test_reference_fixture_is_well_formed() -> None:
    path = Path(__file__).resolve().parents[2] / "regression/cfc_official_nmae_eccv22.json"
    data = json.loads(path.read_text())

    assert set(data["results"]) == {"baseline", "baseline++"}
    assert sum(len(clips) for clips in data["results"]["baseline"].values()) == 1085
