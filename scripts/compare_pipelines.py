"""Compare pipeline variants on identical cached detections (one-factor ablations).

Runs every variant of a plan (``configs/tracking/tuning/*-plan-*.yaml``), reports counting
nMAE and passage-level errors, selects by nMAE (ties keep the current pipeline), and
compares the selected variant with the current one by a paired clip bootstrap. The test
partition cannot be selected.

Example:
    uv run python scripts/compare_pipelines.py \\
        --plan configs/tracking/tuning/suppression-plan-1.yaml \\
        --detections runs/neural/yolox-tiny-v1/epoch-025-7c9bfedc/val \\
        --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from passagewatch.evaluation.ablation import evaluate_variant, plan_variants, select
from passagewatch.evaluation.bootstrap import paired_bootstrap
from passagewatch.inference.batch import make_layout, select_rows
from passagewatch.inference.neural import load_detections
from passagewatch.ingestion.cfc import load_clip

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True, help="cache directory")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--partition", choices=["train", "val"], default="val")
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    variants = plan_variants(plan, REPO_ROOT)
    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = select_rows(manifest_path, [args.partition])
    layout = make_layout(args.manifest.split("-")[0], args.extract_dir, args.frames_dir, rows)
    metadata = {loc: layout.metadata(loc).clips for loc in {r["location"] for r in rows}}
    clips = [
        load_clip(layout, r["location"], metadata[r["location"]][r["clip_name"]]) for r in rows
    ]
    cached = {
        c.name: load_detections(args.detections / c.location / f"{c.name}.npz")[1] for c in clips
    }

    results = []
    for variant in variants:
        r = evaluate_variant(clips, cached, variant)
        results.append(r)
        ref, pred = r.passages["reference_passages"], r.passages["predicted_passages"]
        print(
            f"  {variant.name:18} nMAE {r.nmae:.4f}  count errors {r.absolute_error:3d}  "
            f"passage errors {r.passage_errors:3d}  (missed {ref['missed_fish']}, merged "
            f"{ref['merged_track']}, split {ref['split_track']}, duplicate {pred['duplicate']}, "
            f"background {pred['background']})"
        )

    current, best = results[0], select(results)
    reference, current_counts = current.counts()
    comparison = paired_bootstrap(reference, current_counts, best.counts()[1])
    d = comparison.difference
    print(f"selected: {best.variant.name}")
    print(
        f"{best.variant.name} - current: {d.estimate:+.4f} [{d.low:+.4f}, {d.high:+.4f}] "
        f"(95% CI, paired clip bootstrap); P(better) = {comparison.probability_b_better:.3f}"
    )

    out = args.out or args.detections.parent / f"{plan['name']}-{args.partition}.json"
    out.write_text(
        json.dumps(
            {
                "plan": plan,
                "detections": str(args.detections),
                "partition": args.partition,
                "results": [
                    {
                        "variant": r.variant.model_dump(mode="json"),
                        "nmae": r.nmae,
                        "count_errors": r.absolute_error,
                        "passage_errors": r.passage_errors,
                        **r.passages,
                        "clip_names": [e.clip_name for e in r.errors],
                        "reference": r.counts()[0],
                        "predicted": r.counts()[1],
                    }
                    for r in results
                ],
                "selected": best.variant.name,
                "selected_minus_current": {
                    "estimate": d.estimate,
                    "low": d.low,
                    "high": d.high,
                    "probability_selected_better": comparison.probability_b_better,
                },
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
