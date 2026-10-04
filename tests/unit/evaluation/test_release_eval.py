from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from passagewatch.evaluation.release_eval import frames_problem, plan_day_chunks, release_rows
from passagewatch.ingestion.metadata import ClipMetadata
from passagewatch.ingestion.subsets import ANY_TOP_DIRECTORY, CfcFrameSelector, archive_locations


def meta(name: str, frames: int) -> ClipMetadata:
    return ClipMetadata(
        clip_name=name,
        num_frames=frames,
        framerate=10.0,
        width=100,
        height=200,
        x_meter_start=0.0,
        x_meter_stop=1.0,
        y_meter_start=1.0,
        y_meter_stop=0.0,
    )


def test_days_are_chunked_in_order_to_fit_the_budget() -> None:
    clips = {
        n: meta(n, f)
        for n, f in [
            ("a_2018-05-26_010000_0_100", 100),
            ("b_2018-05-26_020000_0_100", 100),
            ("c_2018-05-27_010000_0_300", 300),
            ("d_2018-05-28_010000_0_50", 50),
            ("e_2018-05-29_010000_0_900", 900),
        ]
    }

    chunks = plan_day_chunks(clips, bytes_per_frame=1.0, budget_bytes=500)

    d = dt.date
    assert chunks == [
        (d(2018, 5, 26), d(2018, 5, 27)),  # 200 + 300
        (d(2018, 5, 28),),  # 50; adding the next day would exceed the budget
        (d(2018, 5, 29),),  # 900: larger than the budget, alone
    ]


def test_releases_are_evaluated_on_test_or_holdout_only() -> None:
    with pytest.raises(ValueError, match="not 'val'"):
        release_rows(Path("unused.parquet"), "val", "kenai-val")


def test_incomplete_frames_are_reported(tmp_path: Path) -> None:
    clip = tmp_path / "clip"
    assert frames_problem(clip, 3) == "no_frames"
    clip.mkdir()
    for i in (0, 2):
        (clip / f"{i}.jpg").write_bytes(b"x")
    assert frames_problem(clip, 3) == "missing_frames:1"
    (clip / "1.jpg").write_bytes(b"x")
    assert frames_problem(clip, 3) is None


def test_test_archives_map_under_any_top_directory() -> None:
    clip = "2018-08-16-JD228_Channel_Stratum1_Set1_CH_2018-08-16_060006_532_732"
    selector = CfcFrameSelector(ANY_TOP_DIRECTORY, {clip: "kenai-channel"}, {clip})

    assert str(selector(f"channel/{clip}_7.jpg")) == f"kenai-channel/{clip}/7.jpg"
    assert selector(f"{clip}_7.jpg") is None  # no top directory
    assert selector(f"channel/sub/{clip}_7.jpg") is None
    assert archive_locations("g945x-41103/rightbank.tar") == ("kenai-rightbank",)
    with pytest.raises(ValueError, match="unknown CFC image archive"):
        archive_locations("g945x-41103/other.tar")
