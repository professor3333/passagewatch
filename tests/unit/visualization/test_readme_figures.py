"""The README's figures are current, well formed, and draw the committed numbers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from passagewatch.visualization.readme_figures import (
    BenchmarkResults,
    all_figures,
    load_test_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION = REPO_ROOT / "releases/evaluations/passagewatch-0.3.0-test.json"
CFC_REFERENCE = REPO_ROOT / "tests/regression/cfc_official_nmae_eccv22.json"
IMAGES = REPO_ROOT / "docs/images"


@pytest.fixture(scope="module")
def results() -> BenchmarkResults:
    return load_test_results(EVALUATION, CFC_REFERENCE)


def test_committed_figures_match_the_generator(results: BenchmarkResults) -> None:
    stale = [
        name
        for name, svg in all_figures(results).items()
        if not (IMAGES / name).exists() or (IMAGES / name).read_text() != svg
    ]
    assert not stale, f"run scripts/make_readme_figures.py: {stale}"


def test_figures_are_well_formed_svg(results: BenchmarkResults) -> None:
    for name, svg in all_figures(results).items():
        root = ET.fromstring(svg)
        assert root.tag == "{http://www.w3.org/2000/svg}svg", name
        assert root.find("{http://www.w3.org/2000/svg}title") is not None, name


def test_every_figure_the_readme_uses_exists() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    for name in (
        "banner.svg",
        "pipeline-light.svg",
        "pipeline-dark.svg",
        "service-light.svg",
        "service-dark.svg",
        "counting-rule-light.svg",
        "counting-rule-dark.svg",
        "test-results-light.svg",
        "test-results-dark.svg",
    ):
        assert f"docs/images/{name}" in readme, name
        assert (IMAGES / name).is_file(), name


def test_numbers_come_from_the_committed_results(results: BenchmarkResults) -> None:
    by_name = {loc.name: loc for loc in results.locations}
    assert results.release == "passagewatch-0.3.0"
    assert (results.clips, results.passages) == (1021, 5296)
    # The release and the classical baseline, as recorded by the test evaluation.
    assert by_name["kenai-rightbank"].nmae["release"] == pytest.approx(0.0746)
    assert results.macro["release"] == pytest.approx(0.2132, abs=1e-4)
    assert results.macro["classical"] == pytest.approx(0.3875, abs=1e-4)
    # CFC's methods, from their published tracks scored by CFC's evaluator.
    assert by_name["elwha"].nmae["cfc_baseline"] == pytest.approx(108 / 334)
    assert by_name["nushagak"].nmae["cfc_baseline_pp"] == pytest.approx(234 / 2654)
    assert results.macro["cfc_baseline"] == pytest.approx(0.2778, abs=1e-4)
    assert results.macro["cfc_baseline_pp"] == pytest.approx(0.1149, abs=1e-4)


def test_the_chart_labels_each_value(results: BenchmarkResults) -> None:
    svg = all_figures(results)["test-results-light.svg"]
    for label in ("16.8", "25.6", "7.5", "35.5", "21.3", "38.8", "27.8", "11.5"):
        assert f">{label}</text>" in svg, label
