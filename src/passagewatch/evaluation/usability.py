"""Analysis of the usability study, by the rules declared in ``docs/usability_study.md``.

- **Time.** Per participant, the ratio of total assisted to total manual active time over
  their scored clips; the time saving is 1 - ratio. Each ratio gets a bootstrap interval
  over that participant's clips (each condition's clips resampled separately: they are
  different clips). With 3 or more **independent** participants, the geometric mean of
  their ratios is reported with an interval from a two-stage bootstrap (participants, then
  their clips). The developer (code ``D...``) is reported alone and never enters the group.
- **Counts.** nMAE per condition, sum(|R^ - R| + |L^ - L|) / sum(R + L), per participant
  and pooled; the difference assisted - manual with intervals from the same resamples.
- **Target met** when the time saving is at least 30% and assisted nMAE is not higher than
  manual nMAE (point estimates); per participant, and for the independent group.
- Practice clips and clips excluded for a recorded technical fault are left out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

TARGET_SAVING = 0.30
MIN_GROUP = 3  # independent participants
DEVELOPER_PREFIX = "D"
RESAMPLES = 10_000


@dataclass(frozen=True)
class Trial:
    participant: str
    condition: str  # manual | assisted
    clip: str
    active_s: float
    final: tuple[int, int]  # right, left
    reference: tuple[int, int]
    automatic: tuple[int, int]

    @property
    def error(self) -> int:
        return abs(self.final[0] - self.reference[0]) + abs(self.final[1] - self.reference[1])

    @property
    def passages(self) -> int:
        return sum(self.reference)

    def fixed_and_introduced(self) -> tuple[int, int]:
        """Per direction, how much closer to (or further from) the reference than automatic."""
        fixed = introduced = 0
        for auto, final, ref in zip(self.automatic, self.final, self.reference, strict=True):
            change = abs(auto - ref) - abs(final - ref)
            fixed += max(change, 0)
            introduced += max(-change, 0)
        return fixed, introduced


def trials_from_rows(
    rows: list[dict[str, str]], plan: dict[str, Any], excluded: set[tuple[str, str]]
) -> list[Trial]:
    """Scored trials: the study's ``trials.csv`` rows joined with the plan's counts."""
    trials = []
    for row in rows:
        if row["practice"] == "1" or (row["participant"], row["clip"]) in excluded:
            continue
        clip = plan["clips"][row["clip"]]
        trials.append(
            Trial(
                participant=row["participant"],
                condition=row["condition"],
                clip=row["clip"],
                active_s=int(row["active_ms"]) / 1000,
                final=(int(row["final_right"]), int(row["final_left"])),
                reference=(clip["reference"][0], clip["reference"][1]),
                automatic=(clip["automatic"][0], clip["automatic"][1]),
            )
        )
    return trials


def nmae(trials: list[Trial]) -> float:
    passages = sum(t.passages for t in trials)
    return sum(t.error for t in trials) / passages if passages else math.nan


def _ratio(manual: list[Trial], assisted: list[Trial]) -> float:
    return sum(t.active_s for t in assisted) / sum(t.active_s for t in manual)


def _interval(values: list[float]) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def _resample(trials: list[Trial], rng: np.random.Generator) -> list[Trial]:
    return [trials[i] for i in rng.integers(len(trials), size=len(trials))]


def _split(trials: list[Trial]) -> tuple[list[Trial], list[Trial]]:
    return [t for t in trials if t.condition == "manual"], [
        t for t in trials if t.condition == "assisted"
    ]


def participant_result(trials: list[Trial], seed: int = 0) -> dict[str, Any]:
    manual, assisted = _split(trials)
    if not manual or not assisted:
        raise ValueError("a participant needs scored clips in both conditions")
    rng = np.random.default_rng(seed)
    ratios, diffs = [], []
    for _ in range(RESAMPLES):
        m, a = _resample(manual, rng), _resample(assisted, rng)
        ratios.append(_ratio(m, a))
        diffs.append(nmae(a) - nmae(m))
    ratio = _ratio(manual, assisted)
    manual_nmae, assisted_nmae = nmae(manual), nmae(assisted)
    fixed = sum(t.fixed_and_introduced()[0] for t in assisted)
    introduced = sum(t.fixed_and_introduced()[1] for t in assisted)
    return {
        "clips": {"manual": len(manual), "assisted": len(assisted)},
        "active_s": {
            "manual": sum(t.active_s for t in manual),
            "assisted": sum(t.active_s for t in assisted),
        },
        "time_ratio": ratio,
        "time_ratio_ci": _interval(ratios),
        "time_saving": 1 - ratio,
        "time_saving_ci": [1 - r for r in reversed(_interval(ratios))],
        "nmae": {"manual": manual_nmae, "assisted": assisted_nmae},
        "nmae_difference": assisted_nmae - manual_nmae,
        "nmae_difference_ci": _interval([d for d in diffs if not math.isnan(d)]),
        "automatic_nmae_on_assisted_clips": nmae(
            [
                Trial(
                    t.participant,
                    t.condition,
                    t.clip,
                    t.active_s,
                    t.automatic,
                    t.reference,
                    t.automatic,
                )
                for t in assisted
            ]
        ),
        "errors_fixed": fixed,
        "errors_introduced": introduced,
        "target_met": 1 - ratio >= TARGET_SAVING and assisted_nmae <= manual_nmae,
    }


def is_developer(participant: str) -> bool:
    return participant.startswith(DEVELOPER_PREFIX)


def group_result(trials: list[Trial], seed: int = 0) -> dict[str, Any] | None:
    """Group results over the independent participants (the developer left out).

    None with fewer than 3 of them: results are then reported per person, as a pilot.
    """
    trials = [t for t in trials if not is_developer(t.participant)]
    people = sorted({t.participant for t in trials})
    if len(people) < MIN_GROUP:
        return None
    by_person = {p: _split([t for t in trials if t.participant == p]) for p in people}

    def summary(groups: list[tuple[list[Trial], list[Trial]]]) -> tuple[float, float]:
        log_ratio = float(np.mean([math.log(_ratio(m, a)) for m, a in groups]))
        manual = [t for m, _ in groups for t in m]
        assisted = [t for _, a in groups for t in a]
        return math.exp(log_ratio), nmae(assisted) - nmae(manual)

    ratio, diff = summary(list(by_person.values()))
    rng = np.random.default_rng(seed)
    ratios, diffs = [], []
    for _ in range(RESAMPLES):
        chosen = [people[i] for i in rng.integers(len(people), size=len(people))]
        groups = [
            (_resample(by_person[p][0], rng), _resample(by_person[p][1], rng)) for p in chosen
        ]
        r, d = summary(groups)
        ratios.append(r)
        if not math.isnan(d):
            diffs.append(d)
    manual = [t for t in trials if t.condition == "manual"]
    assisted = [t for t in trials if t.condition == "assisted"]
    return {
        "participants": people,
        "time_ratio_geometric_mean": ratio,
        "time_ratio_ci": _interval(ratios),
        "time_saving": 1 - ratio,
        "time_saving_ci": [1 - r for r in reversed(_interval(ratios))],
        "nmae": {"manual": nmae(manual), "assisted": nmae(assisted)},
        "nmae_difference": diff,
        "nmae_difference_ci": _interval(diffs),
        "target_met": 1 - ratio >= TARGET_SAVING and diff <= 0,
    }


def sus_score(answers: list[int]) -> float:
    """SUS, 0-100: odd items (answer - 1), even items (5 - answer), the sum times 2.5."""
    return 2.5 * sum(a - 1 if i % 2 == 0 else 5 - a for i, a in enumerate(answers))


def analyze(
    trials: list[Trial], questionnaires: list[dict[str, Any]], seed: int = 0
) -> dict[str, Any]:
    people = sorted({t.participant for t in trials})
    forms: dict[str, dict[str, Any]] = {}
    for q in questionnaires:
        forms.setdefault(q["participant"], {})[q["condition"]] = {
            "sus": sus_score(q["sus"]),
            "tlx_raw": float(np.mean(q["tlx"])),
        }
    return {
        "participants": {
            p: {
                "role": "developer" if is_developer(p) else "independent",
                **participant_result([t for t in trials if t.participant == p], seed),
                "questionnaires": forms.get(p, {}),
            }
            for p in people
        },
        "group": group_result(trials, seed),
        "rules": {"target_saving": TARGET_SAVING, "min_group": MIN_GROUP, "resamples": RESAMPLES},
    }
