from __future__ import annotations

import pytest

from passagewatch.evaluation.bootstrap import paired_bootstrap


def test_point_estimates_are_the_plain_nmae() -> None:
    reference = [[3, 1], [0, 4], [2, 2]]
    a = [[2, 1], [1, 6], [2, 2]]  # errors 1 + 3 + 0 = 4 over 12 passages
    b = [[3, 1], [0, 4], [2, 1]]  # error 1

    result = paired_bootstrap(reference, a, b, resamples=2000)

    assert result.a.estimate == pytest.approx(4 / 12)
    assert result.b.estimate == pytest.approx(1 / 12)
    assert result.difference.estimate == pytest.approx(-3 / 12)
    assert result.a.low <= result.a.estimate <= result.a.high


def test_identical_systems_have_a_zero_difference() -> None:
    reference = [[5, 0], [0, 3], [1, 1], [4, 2]]
    predicted = [[4, 0], [0, 3], [2, 1], [4, 0]]

    result = paired_bootstrap(reference, predicted, predicted, resamples=1000)

    assert result.difference.low == result.difference.high == 0.0
    assert result.probability_b_better == 0.0


def test_a_consistently_better_system_has_an_interval_below_zero() -> None:
    reference = [[5, 5]] * 30
    a = [[3, 7]] * 30  # error 4 in every clip
    b = [[5, 4]] * 30  # error 1 in every clip

    result = paired_bootstrap(reference, a, b, resamples=1000)

    assert result.difference.high < 0
    assert result.probability_b_better == 1.0


def test_resampling_is_reproducible_and_handles_empty_resamples() -> None:
    reference = [[1, 0], [0, 0], [0, 0]]
    a = [[0, 0], [1, 0], [0, 0]]
    b = [[1, 0], [0, 0], [0, 1]]

    first = paired_bootstrap(reference, a, b, resamples=500, seed=7)
    again = paired_bootstrap(reference, a, b, resamples=500, seed=7)

    assert first == again
    assert first.skipped > 0 and first.resamples + first.skipped == 500


def test_no_true_passages_is_an_error() -> None:
    with pytest.raises(ValueError, match="undefined"):
        paired_bootstrap([[0, 0]], [[1, 0]], [[0, 0]])
