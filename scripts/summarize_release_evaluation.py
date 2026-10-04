"""Summarize a declared release's evaluation: per-location and macro-average nMAE.

Reads the per-location files written by ``evaluate_release.py`` and reports, for each system
(``release``, ``baseline``, ``classical``):

- directional nMAE per location with a clip-bootstrap 95% interval, and the macro average
  over locations (the headline) with a bootstrap stratified by location;
- paired differences against the release, on the same clips and resamples;
- the runtime parity check (ONNX on the CPU against the evaluated runtime);
- quarantined clips with their reasons.

A location with no true passages has no nMAE; its absolute error and false counts are
reported instead. Writes ``releases/evaluations/<release>-<partition>.json`` (refusing to
overwrite a different result).

Example:
    uv run python scripts/summarize_release_evaluation.py --results release-eval/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("release", "baseline", "classical")
RESAMPLES = 10_000


def nmae(reference: np.ndarray, predicted: np.ndarray) -> float | None:
    passages = reference.sum()
    return float(np.abs(predicted - reference).sum() / passages) if passages else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    files = sorted(args.results.glob("*.json"))
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    if not reports:
        parser.error(f"no results in {args.results}")
    release = {r["release"] for r in reports}
    partition = {r["partition"] for r in reports}
    if len(release) != 1 or len(partition) != 1:
        parser.error("the results mix releases or partitions")
    systems = [s for s in SYSTEMS if all(s in c for r in reports for c in r["clips"].values())]

    rng = np.random.default_rng(0)
    locations: dict[str, Any] = {}
    samples: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    for report in reports:
        names = sorted(report["clips"])
        reference = np.array([report["clips"][n]["reference"] for n in names])
        predicted = {
            s: np.array([report["clips"][n][s]["predicted"] for n in names]) for s in systems
        }
        idx = rng.integers(0, len(names), size=(RESAMPLES, len(names)))
        ref_passages = reference.sum(axis=1)
        entry: dict[str, Any] = {
            "clips": len(names),
            "passages": int(reference.sum()),
            "quarantined": report["quarantined"],
        }
        for s in systems:
            errors = np.abs(predicted[s] - reference).sum(axis=1)
            denominators = ref_passages[idx].sum(axis=1)
            per_resample = np.where(
                denominators > 0, errors[idx].sum(axis=1) / np.maximum(denominators, 1), np.nan
            )
            samples[s].append(per_resample)
            value = nmae(reference, predicted[s])
            entry[s] = {
                "nmae": None if value is None else round(value, 4),
                "nmae_95ci": None
                if value is None
                else [round(float(q), 4) for q in np.nanpercentile(per_resample, [2.5, 97.5])],
                "absolute_error": int(errors.sum()),
                "false_passages": int(np.clip(predicted[s] - reference, 0, None).sum()),
                "missed_passages": int(np.clip(reference - predicted[s], 0, None).sum()),
            }
        parity = [
            (
                report["clips"][n]["release_onnx"]["predicted"],
                report["clips"][n]["release"]["predicted"],
            )
            for n in names
            if "release_onnx" in report["clips"][n]
        ]
        entry["onnx_parity"] = {
            "clips": len(parity),
            "clips_with_different_counts": sum(a != b for a, b in parity),
        }
        locations[report["location"]] = entry

    macro: dict[str, Any] = {}
    for s in systems:
        values = [locations[loc][s]["nmae"] for loc in locations]
        stacked = np.vstack(samples[s])  # (locations, resamples)
        macro[s] = {
            "nmae": None if None in values else round(float(np.mean(values)), 4),
            "nmae_95ci": [
                round(float(q), 4)
                for q in np.nanpercentile(np.nanmean(stacked, axis=0), [2.5, 97.5])
            ],
        }
        if s != "release":
            diff = np.nanmean(np.vstack(samples["release"]), axis=0) - np.nanmean(stacked, axis=0)
            macro[s]["release_minus_this"] = {
                "estimate": None
                if macro[s]["nmae"] is None or "nmae" not in macro.get("release", {})
                else round(macro["release"]["nmae"] - macro[s]["nmae"], 4),
                "95ci": [round(float(q), 4) for q in np.percentile(diff, [2.5, 97.5])],
                "probability_release_better": round(float((diff < 0).mean()), 4),
            }

    summary = {
        "release": release.pop(),
        "partition": partition.pop(),
        "result_files": [str(p) for p in files],
        "resamples": RESAMPLES,
        "locations": locations,
        "macro_average": macro,
    }
    out = (
        args.out
        or REPO_ROOT / "releases/evaluations" / f"{summary['release']}-{summary['partition']}.json"
    )
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if {k: v for k, v in existing.items() if k != "result_files"} != {
            k: v for k, v in summary.items() if k != "result_files"
        }:
            parser.error(f"{out} exists with a different result")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    print(f"{summary['release']} on {summary['partition']} -> {out}")
    header = "  location           clips passages  " + "  ".join(f"{s:>22}" for s in systems)
    print(header)
    for loc, e in locations.items():
        cells = "  ".join(
            f"{'undef' if e[s]['nmae'] is None else e[s]['nmae']:>7} {e[s]['nmae_95ci']!s:>14}"
            for s in systems
        )
        print(f"  {loc:18} {e['clips']:5d} {e['passages']:8d}  {cells}")
    print(
        "  macro average"
        + " " * 22
        + "  ".join(f"{macro[s]['nmae']!s:>7} {macro[s]['nmae_95ci']!s:>14}" for s in systems)
    )
    for s in systems[1:]:
        d = macro[s]["release_minus_this"]
        print(
            f"  release - {s}: {d['estimate']} {d['95ci']} "
            f"(P release better {d['probability_release_better']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
