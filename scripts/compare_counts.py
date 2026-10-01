"""Compare two systems' directional counts on the same clips with a paired clip bootstrap.

Each side is a report written by ``run_classical.py`` (``report.json``), by
``evaluate_neural.py`` (``report-<partition>.json``, with ``--epoch`` and ``--threshold``
selecting one result), or by ``evaluate_counts.py --out`` (e.g. CFC's published tracks).
Both must cover the same clips with identical reference counts.

Example:
    uv run python scripts/compare_counts.py \\
        --a runs/classical/classical-v2-c4717f5f/full-v2/report.json \\
        --b runs/neural/yolox-tiny-v1/report-val.json --b-epoch 25 --b-threshold 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from passagewatch.evaluation.bootstrap import paired_bootstrap


def clip_counts(
    report: Path, epoch: int | None, threshold: float | None
) -> dict[str, tuple[list[int], list[int]]]:
    data: dict[str, Any] = json.loads(report.read_text(encoding="utf-8"))
    if "results" in data:  # evaluate_neural.py report
        matches = [
            r for r in data["results"] if r["epoch"] == epoch and r["threshold"] == threshold
        ]
        if len(matches) != 1:
            raise ValueError(f"{report}: no single result for epoch {epoch}, threshold {threshold}")
        clips = matches[0]["clips"]
    elif "clips" in data:  # run_classical.py report
        clips = data["clips"]
    else:  # evaluate_counts.py report (e.g. CFC's published tracks)
        clips = [c for loc in data["locations"] for c in loc["clip_errors"]]
    return {c["clip_name"]: (c["reference"], c["predicted"]) for c in clips}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--a-epoch", type=int)
    parser.add_argument("--a-threshold", type=float)
    parser.add_argument("--b-epoch", type=int)
    parser.add_argument("--b-threshold", type=float)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)

    a = clip_counts(args.a, args.a_epoch, args.a_threshold)
    b = clip_counts(args.b, args.b_epoch, args.b_threshold)
    if set(a) != set(b):
        parser.error(f"the reports cover different clips ({len(a)} vs {len(b)})")
    names = sorted(a)
    if any(a[n][0] != b[n][0] for n in names):
        parser.error("the reports disagree on reference counts")
    result = paired_bootstrap(
        [a[n][0] for n in names],
        [a[n][1] for n in names],
        [b[n][1] for n in names],
        resamples=args.resamples,
    )
    print(f"{len(names)} clips, {result.resamples} resamples ({result.skipped} skipped), 95% CI")
    for label, iv in (("a", result.a), ("b", result.b), ("b - a", result.difference)):
        print(f"  {label:6} nMAE {iv.estimate:+.4f}  [{iv.low:+.4f}, {iv.high:+.4f}]")
    print(f"  P(b better than a) over resamples: {result.probability_b_better:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
