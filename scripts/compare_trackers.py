"""Compare trackers on identical cached detections, by counting nMAE with a paired bootstrap.

Runs every setting of a tracker plan (``configs/tracking/tuning/tracker-plan-*.yaml``) on
detections cached by ``evaluate_neural.py``, selects the best setting of each tracker by
counting nMAE, and compares the two selected trackers with a paired clip bootstrap. Only
the tracker changes; detector, detections, and counting are identical.

Example:
    uv run python scripts/compare_trackers.py --plan configs/tracking/tuning/tracker-plan-1.yaml \\
        --detections runs/neural/yolox-tiny-v1/epoch-025-7c9bfedc/val \\
        --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1 --partition val
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from passagewatch.evaluation.bootstrap import paired_bootstrap
from passagewatch.evaluation.detection import DetectionMatch
from passagewatch.inference.batch import make_layout, select_rows
from passagewatch.inference.classical import load_classical_config
from passagewatch.inference.neural import above, load_detections, track_and_evaluate
from passagewatch.ingestion.cfc import load_clip
from passagewatch.tracking.bytetrack import ByteTrackConfig
from passagewatch.tracking.kalman import TrackerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def settings(section: dict[str, Any]) -> list[dict[str, Any]]:
    """Either explicit ``settings`` (for coupled parameters) or the product of ``vary``."""
    if "settings" in section:
        return [dict(s) for s in section["settings"]]
    vary: dict[str, list[Any]] = section["vary"]
    names = list(vary)
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*vary.values())]


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

    kalman_base = load_classical_config(REPO_ROOT / plan["kalman"]["base"]).tracker
    kalman_threshold = float(plan["kalman"]["score_threshold"])
    candidates: list[tuple[str, dict[str, Any], TrackerConfig | ByteTrackConfig, float]] = [
        ("kalman", values, kalman_base.model_copy(update=values), kalman_threshold)
        for values in settings(plan["kalman"])
    ]
    byte_base = ByteTrackConfig.model_validate(plan["bytetrack"]["base"])
    candidates += [
        ("bytetrack", values, byte_base.model_copy(update=values), 0.0)
        for values in settings(plan["bytetrack"])
    ]

    results = []
    for kind, values, tracker, threshold in candidates:
        evaluations = [
            track_and_evaluate(c, above(cached[c.name], threshold), tracker) for c in clips
        ]
        reference = [[e.error.reference.right, e.error.reference.left] for e in evaluations]
        predicted = [[e.error.predicted.right, e.error.predicted.left] for e in evaluations]
        errors = sum(e.error.absolute_error for e in evaluations)
        passages = sum(e.error.reference.total for e in evaluations)
        missed = sum(
            max(0, r[0] - p[0]) + max(0, r[1] - p[1])
            for r, p in zip(reference, predicted, strict=True)
        )
        match = sum((e.detection for e in evaluations), DetectionMatch(0, 0, 0))
        results.append(
            {
                "tracker": kind,
                "values": values,
                "config": tracker.model_dump(mode="json"),
                "score_threshold": threshold,
                "nmae": errors / passages,
                "errors": errors,
                "missed": missed,
                "false": errors - missed,
                "tracks": sum(e.tracks for e in evaluations),
                "clip_names": [e.error.clip_name for e in evaluations],
                "reference": reference,
                "predicted": predicted,
                "detection_recall": match.recall,
            }
        )
        r = results[-1]
        print(
            f"  {kind:9} {values}: nMAE {r['nmae']:.4f}  missed {missed:3d}  "
            f"false {r['false']:3d}  tracks {r['tracks']}"
        )

    base_configs = {
        "kalman": kalman_base.model_dump(mode="json"),
        "bytetrack": byte_base.model_dump(mode="json"),
    }

    def best(kind: str) -> dict[str, Any]:
        # On ties, keep the base (current) settings: a change needs a measured benefit.
        return min(
            (r for r in results if r["tracker"] == kind),
            key=lambda r: (r["nmae"], r["errors"], r["config"] != base_configs[kind]),
        )

    kalman, byte = best("kalman"), best("bytetrack")
    comparison = paired_bootstrap(kalman["reference"], kalman["predicted"], byte["predicted"])
    print(f"best kalman    {kalman['values']}: nMAE {kalman['nmae']:.4f}")
    print(f"best bytetrack {byte['values']}: nMAE {byte['nmae']:.4f}")
    d = comparison.difference
    print(
        f"bytetrack - kalman: {d.estimate:+.4f} [{d.low:+.4f}, {d.high:+.4f}] (95% CI, "
        f"paired clip bootstrap); P(bytetrack better) = {comparison.probability_b_better:.3f}"
    )

    out = args.out or args.detections.parent / f"trackers-{plan['name']}-{args.partition}.json"
    out.write_text(
        json.dumps(
            {
                "plan": plan,
                "detections": str(args.detections),
                "partition": args.partition,
                "results": results,
                "best": {"kalman": kalman["values"], "bytetrack": byte["values"]},
                "bytetrack_minus_kalman": {
                    "estimate": d.estimate,
                    "low": d.low,
                    "high": d.high,
                    "probability_bytetrack_better": comparison.probability_b_better,
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
