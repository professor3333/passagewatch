"""Our counting evaluator must reproduce CFC's official nMAE exactly, clip by clip.

The reference was produced by running CFC's own ``evaluate.py`` (with its pinned TrackEval)
on the published ECCV22 Baseline and Baseline++ tracks; see ``docs/counting_policy.md`` §5.
Those are CFC's predictions, not PassageWatch's, so this uses no test-set information for
tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from passagewatch.evaluation.nmae import evaluate_mot_location, summarize
from passagewatch.ingestion.cfc import CfcLayout

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED = REPO_ROOT / "data/extracted/cfc"
RESULTS = EXTRACTED / "fish_counting_results/results"
REFERENCE = json.loads((Path(__file__).parent / "cfc_official_nmae_eccv22.json").read_text())

CASES = [
    (tracker, location)
    for tracker, locations in REFERENCE["results"].items()
    for location in locations
]


@pytest.mark.slow
@pytest.mark.skipif(not RESULTS.is_dir(), reason="CFC labels bundle not present")
@pytest.mark.parametrize(("tracker", "location"), CASES)
def test_matches_official_nmae_per_clip(tracker: str, location: str) -> None:
    layout = CfcLayout.full(EXTRACTED)
    expected = REFERENCE["results"][tracker][location]

    errors = evaluate_mot_location(
        layout.annotations_dir / location,
        RESULTS / location / tracker / "data",
        layout.metadata(location).clips,
    )

    actual = {e.clip_name: [e.absolute_error, e.reference.total] for e in errors}
    assert actual == expected
    total = summarize(location, errors)
    assert total.absolute_error == sum(v[0] for v in expected.values())
    assert total.reference_passages == sum(v[1] for v in expected.values())
