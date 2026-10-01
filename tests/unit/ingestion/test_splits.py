from __future__ import annotations

import datetime as dt

import pytest

from passagewatch.ingestion.cfc import LOCATIONS
from passagewatch.ingestion.splits import (
    OFFICIAL_SPLITS,
    Split,
    official_split,
    parse_clip_name,
    tuning_allowed,
)


@pytest.mark.parametrize(
    ("name", "recording", "date", "start", "stop"),
    [
        (
            "2018-05-27-JD147_LeftFar_Stratum2_Set1_LO_2018-05-27_091000_907_1387",
            "2018-05-27-JD147_LeftFar_Stratum2_Set1_LO_2018-05-27_091000",
            dt.date(2018, 5, 27),
            907,
            1387,
        ),
        (
            "Elwha_2018_OM_ARIS_2018_07_09_2018-07-09_190000_490_941",
            "Elwha_2018_OM_ARIS_2018_07_09_2018-07-09_190000",
            dt.date(2018, 7, 9),
            490,
            941,
        ),
        (
            "RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000_3600_3900",
            "RB_Nusagak_Sonar_Files_2018_RB_2018-07-02_211000",
            dt.date(2018, 7, 2),
            3600,
            3900,
        ),
    ],
)
def test_clip_names_split_into_recording_and_frame_range(
    name: str, recording: str, date: dt.date, start: int, stop: int
) -> None:
    parsed = parse_clip_name(name)

    assert parsed.recording_id == recording
    assert parsed.recording_date == date
    assert (parsed.start, parsed.stop) == (start, stop)


@pytest.mark.parametrize(
    "name", ["clip", "cam_2018-06-01_120000_100", "cam_2018-06-01_120000_300_100"]
)
def test_malformed_clip_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="clip name"):
        parse_clip_name(name)


def test_every_location_has_an_official_split() -> None:
    assert set(OFFICIAL_SPLITS) == set(LOCATIONS)
    assert official_split("kenai-train") is Split.TRAIN
    assert official_split("kenai-val") is Split.VAL
    assert {loc for loc in LOCATIONS if official_split(loc) is Split.TEST} == {
        "kenai-rightbank",
        "kenai-channel",
        "elwha",
        "nushagak",
    }


def test_test_locations_are_never_tunable() -> None:
    assert tuning_allowed(Split.TRAIN)
    assert tuning_allowed(Split.VAL)
    assert not tuning_allowed(Split.TEST)


def test_unknown_location_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown CFC location"):
        official_split("kenai-left")
