"""Tune the classical baseline one factor at a time, by counting nMAE on a development partition.

The plan file lists experiments in order. Each experiment varies one or two parameters of
the current best configuration over a few values; the value with the lowest nMAE (macro
over locations, then fewest errors) becomes the new best before the next experiment
starts. Every run is recorded in ``tuning.json``, so the search and its budget are visible.

Example:
    uv run python scripts/tune_classical.py --plan configs/tracking/tuning/classical-plan-1.yaml \\
        --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1 --partition train
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from passagewatch.inference.batch import (
    default_workers,
    make_layout,
    make_tasks,
    run_tasks,
    select_rows,
    summarize_outcomes,
)
from passagewatch.inference.classical import ClassicalConfig, load_classical_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def with_values(config: ClassicalConfig, values: dict[str, Any], name: str) -> ClassicalConfig:
    data = config.model_dump()
    for dotted, value in values.items():
        section, key = dotted.split(".")
        if key not in data[section]:
            raise KeyError(f"unknown parameter {dotted}")
        data[section][key] = value
    data["name"] = name
    return ClassicalConfig.model_validate(data)


def score(summary: dict[str, Any]) -> tuple[float, int]:
    macro = summary["macro_nmae"]
    errors = sum(g["absolute_error"] for g in summary["locations"])
    return (float("inf") if macro is None else macro, errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--partition", choices=["train", "val"], required=True)
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    best = load_classical_config(REPO_ROOT / plan["base"])
    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = select_rows(manifest_path, [args.partition])
    layout = make_layout(args.manifest.split("-")[0], args.extract_dir, args.frames_dir, rows)
    out = args.out or REPO_ROOT / "runs/tuning" / f"{plan['name']}-{args.manifest}-{args.partition}"
    out.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}
    log: list[dict[str, Any]] = []

    def evaluate(config: ClassicalConfig) -> dict[str, Any]:
        key = config.model_copy(update={"name": "x"}).sha256()
        if key not in cache:
            outcomes = run_tasks(make_tasks(rows, layout, config, None), args.workers)
            cache[key] = summarize_outcomes(outcomes)
        return cache[key]

    baseline = evaluate(best)
    best_summary = baseline
    print(f"base {best.name}: macro nMAE {baseline['macro_nmae']}", flush=True)
    for experiment in plan["experiments"]:
        grid = experiment["vary"]
        names = list(grid)
        trials = []
        for combo in itertools.product(*(grid[n] for n in names)):
            values = dict(zip(names, combo, strict=True))
            candidate = with_values(best, values, best.name)
            summary = evaluate(candidate)
            trials.append((score(summary), values, candidate, summary))
            nmae = summary["macro_nmae"]
            print(f"  {experiment['id']} {values}: macro nMAE {nmae:.4f}", flush=True)
            log.append(
                {
                    "experiment": experiment["id"],
                    "values": values,
                    "macro_nmae": nmae,
                    "locations": summary["locations"],
                    "ms_per_frame": summary["runtime"]["ms_per_frame"],
                }
            )
        trials.sort(key=lambda t: t[0])
        if trials[0][0] < score(best_summary):
            _, values, best, best_summary = trials[0]
            print(f"{experiment['id']}: keep {values}", flush=True)
            edges = [
                n for n in names if len(grid[n]) > 2 and values[n] in (grid[n][0], grid[n][-1])
            ]
            if edges:
                # The optimum may lie outside the grid; the plan should be widened.
                print(
                    f"{experiment['id']}: WARNING best value at grid edge for {edges}", flush=True
                )
                log.append({"experiment": experiment["id"], "grid_edge": edges})
        else:
            print(f"{experiment['id']}: no improvement; keep current values", flush=True)

    final = best.model_copy(update={"name": plan["result_name"]})
    (out / "tuning.json").write_text(
        json.dumps(
            {
                "plan": plan,
                "manifest": args.manifest,
                "partition": args.partition,
                "clips": len(rows),
                "runs": len(cache),
                "base_macro_nmae": baseline["macro_nmae"],
                "final_macro_nmae": best_summary["macro_nmae"],
                "final_config": final.model_dump(mode="json"),
                "log": log,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / f"{final.name}.yaml").write_text(
        yaml.safe_dump(final.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    print(
        f"final {final.name}: macro nMAE {best_summary['macro_nmae']} ({len(cache)} runs) -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
