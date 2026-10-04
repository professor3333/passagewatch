"""Measure review prioritization on a development partition (Stage 9).

For a neural detector's cached detections, tracks every clip, scores every trajectory
(``review-score-v0``), labels counting errors with the one-to-one error analysis, and
reports the share of errors a reviewer finds within a share of the review time, for three
orders: triage then review score (the queue), review score alone, and random order. It also
reports how errors spread over the triage states, and how many outright-missed fish fall in
random audit windows. The test partition cannot be selected.

Example:
    uv run python scripts/evaluate_review.py \\
        --detections runs/neural/yolox-tiny-v1/epoch-025-7c9bfedc/val \\
        --manifest full-v2 --frames-dir data/extracted/cfc/kenai-dev-v1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from passagewatch.calibration.audit import AUDIT_VERSION, seed_from
from passagewatch.calibration.review_score import REVIEW_SCORE_VERSION
from passagewatch.calibration.versions import CALIBRATIONS, REVIEW_V0, review_clip
from passagewatch.counting.policy import CFC_COMPATIBLE_V1, count_trajectories
from passagewatch.evaluation.errors import analyze_clip
from passagewatch.evaluation.prioritization import (
    ErrorSet,
    QueueItem,
    found_curve,
    ordered,
    queue_items,
    random_order_curve,
)
from passagewatch.inference.batch import make_layout, select_rows
from passagewatch.inference.classical import load_classical_config
from passagewatch.inference.neural import above, load_detections, track_clip
from passagewatch.ingestion.cfc import load_clip
from passagewatch.tracking.kalman import trajectories_to_annotations

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
TRIAGE_RANK = {"unresolved": 0, "needs_review": 1, "suggested": 2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detections", type=Path, required=True, help="cache directory")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--partition",
        choices=["train", "val", "holdout"],
        default="val",
        help="holdout: evaluates fixed review settings, never selects them",
    )
    parser.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "data/extracted/cfc")
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--tracking-config", type=Path, default=REPO_ROOT / "configs/tracking/classical-v2.yaml"
    )
    parser.add_argument("--calibration", choices=sorted(CALIBRATIONS), default=REVIEW_V0)
    parser.add_argument(
        "--suggest-threshold",
        type=float,
        default=None,
        help="try another suggest_threshold than the calibration version's (an experiment)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest_path = REPO_ROOT / f"data/manifests/splits/cfc/{args.manifest}.parquet"
    rows = select_rows(manifest_path, [args.partition])
    layout = make_layout(args.manifest.split("-")[0], args.extract_dir, args.frames_dir, rows)
    metadata = {loc: layout.metadata(loc).clips for loc in {r["location"] for r in rows}}
    tracker = load_classical_config(args.tracking_config).tracker
    policy = CFC_COMPATIBLE_V1
    calibration = CALIBRATIONS[args.calibration]
    if args.suggest_threshold is not None:
        review_config = calibration.review.model_copy(
            update={"suggest_threshold": args.suggest_threshold}
        )
        calibration = dataclasses.replace(
            calibration,
            version=f"{calibration.version}+suggest-{args.suggest_threshold}",
            review=review_config,
        )

    items: list[QueueItem] = []
    reachable: set[str] = set()
    unreachable: set[str] = set()
    keys: dict[tuple[str, int], tuple[int, int, float]] = {}
    scores: dict[tuple[str, int], float] = {}
    triage_errors: Counter[tuple[str, bool]] = Counter()
    audited_missed = 0
    audit_frames = total_frames = 0
    for row in rows:
        meta = metadata[row["location"]][row["clip_name"]]
        clip = load_clip(layout, row["location"], meta)
        _, cached = load_detections(args.detections / row["location"] / f"{clip.name}.npz")
        # Numbered 1..n, as the service does, so that trajectories, counts and the error
        # analysis (which renumbers in the same order) agree on track IDs.
        trajectories = [
            dataclasses.replace(t, track_id=i)
            for i, t in enumerate(track_clip(clip, above(cached, args.threshold), tracker), 1)
        ]
        predicted = trajectories_to_annotations(trajectories)
        counts = count_trajectories(predicted, meta.width, meta.height, policy)
        mpp = (
            abs(meta.x_meter_stop - meta.x_meter_start) / meta.width,
            abs(meta.y_meter_stop - meta.y_meter_start) / meta.height,
        )
        clip_review = review_clip(
            calibration,
            trajectories,
            counts,
            meters_per_px=mpp,
            line_x_normalized=policy.line_x_normalized,
            num_frames=clip.num_window_frames,
            framerate=meta.framerate,
            frame_offset=clip.frame_start,
            seed=seed_from(clip.name),
        )
        reviews = clip_review.tracks
        analysis = analyze_clip(
            clip.name, clip.annotations, predicted, meta.width, meta.height, policy
        )
        spans = {tid: r.features.span_frames for tid, r in reviews.items()}
        clip_items, errors = queue_items(analysis, spans, meta.framerate)
        items.extend(clip_items)
        reachable |= errors.reachable
        unreachable |= errors.unreachable
        for item in clip_items:
            r = reviews[item.track_id]
            key = (clip.name, item.track_id)
            keys[key] = (TRIAGE_RANK[r.triage], 0 if r.features.is_passage else 1, r.review_score)
            scores[key] = r.review_score
            triage_errors[(r.triage, bool(item.reveals))] += 1

        # Random audits of unflagged footage: do they land on the fish nothing tracked?
        windows = clip_review.audit
        audit_frames += sum(w.frames for w in windows)
        total_frames += clip.num_window_frames
        ann = clip.annotations
        for error in errors.unreachable:
            ref_id = int(error.rsplit(":", 1)[1])
            frames = ann.frame_index[ann.track_id == ref_id] - clip.frame_start
            if any(((frames >= w.start_frame) & (frames < w.stop_frame)).any() for w in windows):
                audited_missed += 1

    errors = ErrorSet(frozenset(reachable), frozenset(unreachable))
    curves = {
        "triage_then_score": found_curve(
            ordered(items, lambda i: keys[(i.clip_name, i.track_id)]), errors, BUDGETS
        ),
        "score_only": found_curve(
            ordered(items, lambda i: scores[(i.clip_name, i.track_id)]), errors, BUDGETS
        ),
        "random": random_order_curve(items, errors, BUDGETS),
    }
    report: dict[str, Any] = {
        "calibration_version": calibration.version,
        "review_score_version": REVIEW_SCORE_VERSION,
        "review_config": calibration.review.model_dump(),
        "audit_version": AUDIT_VERSION,
        "detections": str(args.detections),
        "threshold": args.threshold,
        "manifest": args.manifest,
        "partition": args.partition,
        "queue_items": len(items),
        "review_time_s": round(sum(i.cost_s for i in items), 1),
        "errors": {"reachable": len(reachable), "unreachable_missed_fish": len(unreachable)},
        "curves": {
            name: {f"{b:.1f}": {"all": a, "reachable": r} for b, (a, r) in c.items()}
            for name, c in curves.items()
        },
        "triage": {
            state: {
                "tracks": triage_errors[(state, True)] + triage_errors[(state, False)],
                "with_errors": triage_errors[(state, True)],
            }
            for state in TRIAGE_RANK
        },
        "audit": {
            "fraction_of_frames": audit_frames / total_frames if total_frames else 0.0,
            "missed_fish_seen": audited_missed,
            "missed_fish": len(unreachable),
        },
    }
    name = calibration.version if args.suggest_threshold is not None else args.partition
    out = args.out or args.detections.parent / (
        f"review-{args.partition}.json"
        if args.suggest_threshold is None
        else f"review-{args.partition}-{name}.json"
    )
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(
        f"{len(items)} queue items, {report['review_time_s']} s of review; errors: "
        f"{len(reachable)} reachable + {len(unreachable)} missed outright"
    )
    print("share of reachable errors found at a share of review time:")
    print("  budget  " + "  ".join(f"{b:>5.1f}" for b in BUDGETS))
    for name, c in curves.items():
        print(f"  {name:18}" + "  ".join(f"{c[b][1]:5.2f}" for b in BUDGETS))
    for state, t in report["triage"].items():
        print(f"  triage {state:13} tracks {t['tracks']:4d}  with errors {t['with_errors']:3d}")
    a = report["audit"]
    print(
        f"  audit covers {a['fraction_of_frames']:.2%} of frames; it shows "
        f"{a['missed_fish_seen']} of {a['missed_fish']} fish missed outright"
    )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
