from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from passagewatch.ingestion.metadata import load_metadata


def entry(name: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "clip_name": name,
        "num_frames": 451,
        "framerate": 7.508070945739746,
        "width": 629,
        "height": 1251,
        "x_meter_start": -3.7,
        "x_meter_stop": 3.66,
        "y_meter_start": 15.3,
        "y_meter_stop": 0.65,
    }
    base.update(overrides)
    return base


def write(tmp_path: Path, entries: object) -> Path:
    path = tmp_path / "elwha.json"
    path.write_text(json.dumps(entries))
    return path


def test_valid_entries_load(tmp_path: Path) -> None:
    result = load_metadata(write(tmp_path, [entry("a"), entry("b", width=300)]))

    assert result.errors == []
    assert set(result.clips) == {"a", "b"}
    assert result.clips["b"].width == 300


def test_unknown_field_is_rejected_not_ignored(tmp_path: Path) -> None:
    result = load_metadata(write(tmp_path, [entry("a", upstream_direction="left"), entry("b")]))

    assert set(result.clips) == {"b"}
    assert "upstream_direction" in result.errors[0]


@pytest.mark.parametrize(
    "bad", [{"width": 0}, {"num_frames": -1}, {"framerate": 0}, {"height": "tall"}]
)
def test_invalid_values_are_reported_per_entry(tmp_path: Path, bad: dict[str, Any]) -> None:
    result = load_metadata(write(tmp_path, [entry("a", **bad), entry("b")]))

    assert set(result.clips) == {"b"}
    assert result.errors[0].startswith("entry 0 ('a')")


def test_duplicate_clip_names_drop_every_copy(tmp_path: Path) -> None:
    result = load_metadata(write(tmp_path, [entry("a"), entry("a", width=1), entry("b")]))

    assert set(result.clips) == {"b"}
    assert "duplicate clip_name 'a'" in result.errors[0]


def test_non_list_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected a JSON list"):
        load_metadata(write(tmp_path, {"clip_name": "a"}))
